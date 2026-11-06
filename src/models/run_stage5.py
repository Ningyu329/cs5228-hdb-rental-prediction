"""Stage 5: nested temporal model/feature experiments, macro ablations and errors.

Candidate selection uses the two earlier backtests, never the primary holdout.
Tree iteration counts use an inner temporal split, then refit all outer fit rows.
No test prediction or competition submission is generated at this stage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
if __package__ in {None, ""}:
    sys.path.insert(0, str(ROOT))

from src.features.build_stage5_features import (
    COE_FEATURES, STOCK_FEATURES, ENGINEERED_CATEGORICAL, ENGINEERED_NUMERIC,
    add_housing_features, aggregate_macro, attach_macro,
)
from src.models.train_baseline import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET, build_ridge_pipeline, rmse, split_metadata
from src.models.run_location_ablation import HDB_CATEGORICAL, HDB_NUMERIC, SCHOOL_NUMERIC, MALL_NUMERIC, MRT_NUMERIC, validation_splits

SCHEMES = ["main", "housing", "amenities", "coe", "market", "mrt_snapshot"]


def implementation_hashes() -> dict[str, str]:
    paths = [Path(__file__), ROOT / "src/features/build_stage5_features.py", ROOT / "src/models/train_baseline.py"]
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def cache_identity(manifest: dict) -> str:
    keys = [
        "environment", "source_sha256", "tree_iterations_cap", "threads", "random_seed",
        "catboost", "lightgbm", "early_stopping", "macro", "feature_sets",
        "constant_fee_values", "constant_furnished_values", "selection_rule", "implementation_sha256",
    ]
    data = {key: manifest[key] for key in keys}
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def feature_schema(scheme: str) -> tuple[list[str], list[str]]:
    cats, nums = list(CATEGORICAL_FEATURES), list(NUMERIC_FEATURES)
    if scheme != "main":
        cats += HDB_CATEGORICAL + ENGINEERED_CATEGORICAL
        nums += HDB_NUMERIC + ENGINEERED_NUMERIC
    if scheme in {"amenities", "coe", "market", "mrt_snapshot"}:
        nums += SCHOOL_NUMERIC + MALL_NUMERIC
    if scheme in {"coe", "market", "mrt_snapshot"}:
        nums += COE_FEATURES
    if scheme in {"market", "mrt_snapshot"}:
        nums += STOCK_FEATURES
    if scheme == "mrt_snapshot":
        nums += MRT_NUMERIC
    if scheme not in SCHEMES:
        raise ValueError(f"Unknown feature scheme: {scheme}")
    return cats, nums


def load_training(path: Path) -> pd.DataFrame:
    frame = add_housing_features(pd.read_csv(path))
    cats, nums = feature_schema("mrt_snapshot")
    nums = [column for column in nums if column not in COE_FEATURES + STOCK_FEATURES]
    missing = set(cats + nums + [TARGET, "RENT_APPROVAL_DATE"]) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing input columns: {sorted(missing)}")
    for column in cats:
        frame[column] = frame[column].fillna("missing").astype(str)
    for column in nums:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    frame[TARGET] = pd.to_numeric(frame[TARGET], errors="raise")
    if not np.isfinite(frame[TARGET]).all():
        raise ValueError("Rent labels must be finite")
    frame["RENT_APPROVAL_DATE"] = pd.to_datetime(frame["RENT_APPROVAL_DATE"], format="ISO8601", errors="raise")
    return frame


@dataclass
class FittedModel:
    family: str
    model: object
    cats: list[str]
    nums: list[str]
    levels: dict[str, list[str]]

    @property
    def features(self) -> list[str]:
        return self.cats + self.nums

    def matrix(self, frame: pd.DataFrame) -> pd.DataFrame:
        x = frame[self.features].copy()
        for column in self.cats:
            values = x[column].fillna("missing").astype(str)
            if self.family == "lightgbm":
                known = values.where(values.isin(self.levels[column]), pd.NA)
                x[column] = pd.Categorical(known, categories=self.levels[column])
            else:
                x[column] = values
        return x

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.maximum(np.asarray(self.model.predict(self.matrix(frame)), dtype=float), 0)


def spline_pipeline() -> tuple[Pipeline, list[str], list[str]]:
    cats = list(CATEGORICAL_FEATURES)
    # Remove all three global linear calendar fields. The spline is constant
    # outside fitted time support, so a historic growth slope cannot extrapolate.
    numeric = ["FLOOR_AREA_SQM", "LEASE_COMMENCE_DATE", "FLAT_AGE", "RENT_MONTH_SIN", "RENT_MONTH_COS"]
    transform = ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore", min_frequency=2, sparse_output=True), cats),
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("time", Pipeline([("impute", SimpleImputer(strategy="median")), ("spline", SplineTransformer(n_knots=5, degree=2, extrapolation="constant")), ("scale", StandardScaler())]), ["RENT_MONTH_INDEX"]),
    ])
    return Pipeline([("preprocess", transform), ("model", Ridge(alpha=10.0, solver="lsqr", tol=1e-4, fit_intercept=False))]), cats, numeric + ["RENT_MONTH_INDEX"]


def fit_tree(
    family: str, fit: pd.DataFrame, query: pd.DataFrame,
    cats: list[str], nums: list[str], iterations: int, threads: int,
    inner: pd.DataFrame | None = None,
) -> FittedModel:
    levels = {column: fit[column].fillna("missing").astype(str).unique().tolist() for column in cats}
    bundle = FittedModel(family, None, cats, nums, levels)
    x = bundle.matrix(fit)
    if family == "catboost":
        from catboost import CatBoostRegressor
        model = CatBoostRegressor(
            iterations=iterations, depth=6, learning_rate=0.06, l2_leaf_reg=5.0,
            loss_function="RMSE", random_seed=42, thread_count=threads,
            allow_writing_files=False, verbose=False,
        )
        kwargs = {"cat_features": cats}
        if inner is not None:
            kwargs.update(eval_set=(bundle.matrix(inner), inner[TARGET]), early_stopping_rounds=40, use_best_model=True)
        model.fit(x, fit[TARGET], **kwargs)
    elif family == "lightgbm":
        from lightgbm import LGBMRegressor, early_stopping
        model = LGBMRegressor(
            n_estimators=iterations, learning_rate=0.04, num_leaves=31, max_depth=-1,
            min_child_samples=40, reg_lambda=5.0, colsample_bytree=0.9,
            random_state=42, n_jobs=threads, verbosity=-1, deterministic=True, force_col_wise=True,
        )
        kwargs = {"categorical_feature": cats}
        if inner is not None:
            kwargs.update(eval_X=bundle.matrix(inner), eval_y=inner[TARGET], eval_metric="rmse", callbacks=[early_stopping(40, verbose=False)])
        model.fit(x, fit[TARGET], **kwargs)
    else:
        raise ValueError(f"Unsupported tree family: {family}")
    bundle.model = model
    return bundle


def fit_candidate(
    family: str, scheme: str, fit: pd.DataFrame, query: pd.DataFrame,
    monthly: pd.DataFrame, tree_iterations: int, threads: int,
) -> tuple[FittedModel, np.ndarray, dict, pd.DataFrame]:
    start = time.perf_counter()
    query_features = attach_macro(query, monthly, fit.RENT_APPROVAL_DATE.max())
    train_features = attach_macro(fit, monthly)
    if family == "ridge":
        cats, nums = feature_schema("main")
        bundle = FittedModel("ridge", build_ridge_pipeline(), cats, nums, {})
        bundle.model.fit(bundle.matrix(train_features), fit[TARGET])
        metadata = {"iterations": None, "inner_fit_end": None, "inner_validation_start": None, "inner_validation_end": None}
    elif family == "ridge_spline":
        model, cats, nums = spline_pipeline()
        bundle = FittedModel("ridge_spline", model, cats, nums, {})
        model.fit(bundle.matrix(train_features), fit[TARGET])
        metadata = {"iterations": None, "inner_fit_end": None, "inner_validation_start": None, "inner_validation_end": None}
    else:
        cats, nums = feature_schema(scheme)
        periods = fit.RENT_APPROVAL_DATE.dt.to_period("M")
        unique = sorted(periods.unique())
        if len(unique) < 4:
            raise ValueError("At least 4 training months are needed for inner temporal early stopping")
        inner_mask = periods.isin(unique[-3:])
        inner_fit, inner_valid = fit.loc[~inner_mask], fit.loc[inner_mask]
        tune_fit = attach_macro(inner_fit, monthly)
        tune_valid = attach_macro(inner_valid, monthly, inner_fit.RENT_APPROVAL_DATE.max())
        tuned = fit_tree(family, tune_fit, tune_valid, cats, nums, tree_iterations, threads, inner=tune_valid)
        rounds = int(tuned.model.tree_count_ if family == "catboost" else tuned.model.best_iteration_)
        rounds = max(rounds, 1)
        # The outer query/labels never enter early stopping or this refit.
        bundle = fit_tree(family, train_features, query_features, cats, nums, rounds, threads)
        metadata = {"iterations": rounds, "inner_fit_end": inner_fit.RENT_APPROVAL_DATE.max().strftime("%Y-%m"), "inner_validation_start": inner_valid.RENT_APPROVAL_DATE.min().strftime("%Y-%m"), "inner_validation_end": inner_valid.RENT_APPROVAL_DATE.max().strftime("%Y-%m")}
    prediction = bundle.predict(query_features)
    metadata["seconds"] = time.perf_counter() - start
    metadata["feature_count"] = len(bundle.features)
    return bundle, prediction, metadata, query_features


def metrics(actual: pd.Series, prediction: np.ndarray) -> dict:
    error = prediction - actual.to_numpy()
    return {"rmse": rmse(actual, prediction), "mae": float(mean_absolute_error(actual, prediction)), "bias": float(error.mean())}


def cached_candidate(
    cache_dir: Path, split: str, family: str, scheme: str,
    fit: pd.DataFrame, query: pd.DataFrame, monthly: pd.DataFrame,
    tree_iterations: int, threads: int,
) -> tuple[FittedModel | None, np.ndarray, dict, pd.DataFrame | None]:
    """Cache completed independent fits so an interrupted run can resume."""
    import joblib
    directory = cache_dir / split / f"{family}_{scheme}"
    result_path = directory / "result.json"
    if result_path.is_file():
        details = json.loads(result_path.read_text(encoding="utf-8"))
        prediction = np.load(directory / "prediction.npy")
        bundle, frame = None, None
        if split == "temporal":
            payload = joblib.load(directory / "model.joblib")
            bundle = FittedModel(**payload)
            frame = attach_macro(query, monthly, fit.RENT_APPROVAL_DATE.max())
        return bundle, prediction, details, frame
    bundle, prediction, details, frame = fit_candidate(family, scheme, fit, query, monthly, tree_iterations, threads)
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "prediction.npy", prediction)
    if split == "temporal":
        # Save fields rather than a __main__-qualified FittedModel object.
        joblib.dump({"family": bundle.family, "model": bundle.model, "cats": bundle.cats, "nums": bundle.nums, "levels": bundle.levels}, directory / "model.joblib")
    # Commit metadata last; incomplete caches are never treated as complete.
    result_path.write_text(json.dumps(details, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return bundle if split == "temporal" else None, prediction, details, frame if split == "temporal" else None


def select_candidate(results: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    earlier = results.loc[results.split.str.startswith("temporal_backtest_") & ~results.scheme.eq("mrt_snapshot")].copy()
    if earlier.empty:
        raise ValueError("Selection requires earlier temporal backtests")
    earlier["sse"] = earlier.rmse ** 2 * earlier.validation_rows
    table = earlier.groupby(["candidate", "family", "scheme"]).agg(sse=("sse", "sum"), rows=("validation_rows", "sum"), windows=("split", "nunique")).reset_index()
    if not table.windows.eq(earlier.split.nunique()).all():
        raise ValueError("Candidate selection needs matching backtest windows")
    table["pooled_backtest_rmse"] = np.sqrt(table.sse / table.rows)
    table = table.sort_values(["pooled_backtest_rmse", "candidate"]).reset_index(drop=True)
    return str(table.iloc[0].candidate), table


def grouped_errors(frame: pd.DataFrame, predictions: dict[str, np.ndarray]) -> pd.DataFrame:
    view = frame.copy()
    view["AREA_BIN"] = pd.cut(view.FLOOR_AREA_SQM, [-np.inf, 50, 70, 90, 110, 130, 160, np.inf]).astype(str)
    view["AGE_BIN"] = pd.cut(view.FLAT_AGE, [-np.inf, 10, 20, 30, 40, 50, np.inf]).astype("string").fillna("missing")
    rows = []
    for candidate, prediction in predictions.items():
        for field in ["TOWN", "FLAT_TYPE", "FLAT_MODEL", "RENT_APPROVAL_DATE", "REGION", "AREA_BIN", "AGE_BIN", "ADDRESS_MATCHED"]:
            for value, indices in view.groupby(field, dropna=False).groups.items():
                positions = view.index.get_indexer(indices)
                row = {"candidate": candidate, "field": field, "value": str(value), "rows": len(positions)}
                row.update(metrics(view.loc[indices, TARGET], np.asarray(prediction)[positions]))
                rows.append(row)
    return pd.DataFrame(rows)


def permutation_groups() -> dict[str, list[str]]:
    return {
        "time": ["RENT_YEAR", "RENT_MONTH", "RENT_MONTH_INDEX", "RENT_MONTH_SIN", "RENT_MONTH_COS"],
        "housing": ["FLAT_TYPE", "FLAT_MODEL", "FLOOR_AREA_SQM", "FLAT_AGE", "LEASE_COMMENCE_DATE", "LOG_FLOOR_AREA", "FLAT_TYPE_ROOMS", "AREA_PER_ROOM", "FLAT_AGE_BAND"],
        "address": ["TOWN", "BLOCK", "STREET", "BLOCK_ADDRESS"],
        "hdb": HDB_CATEGORICAL + HDB_NUMERIC,
        "amenities": SCHOOL_NUMERIC + MALL_NUMERIC, "coe": COE_FEATURES, "stock": STOCK_FEATURES, "mrt": MRT_NUMERIC,
    }


def group_permutation(bundles: list[FittedModel], query: pd.DataFrame, sample_size: int = 3000) -> pd.DataFrame:
    sample = query.sample(min(len(query), sample_size), random_state=42)
    base = np.mean([bundle.predict(sample) for bundle in bundles], axis=0)
    baseline = rmse(sample[TARGET], base)
    features = set().union(*(bundle.features for bundle in bundles))
    rng = np.random.default_rng(42)
    rows = []
    for group, columns in permutation_groups().items():
        columns = [column for column in columns if column in features]
        if not columns:
            continue
        changes = []
        for _ in range(2):
            shuffled = sample.copy()
            order = rng.permutation(len(sample))
            for column in columns:
                shuffled[column] = sample[column].iloc[order].to_numpy()
            prediction = np.mean([bundle.predict(shuffled) for bundle in bundles], axis=0)
            changes.append(rmse(sample[TARGET], prediction) - baseline)
        rows.append({"group": group, "rmse_increase_mean": float(np.mean(changes)), "rmse_increase_std": float(np.std(changes)), "sample_rows": len(sample), "sample_baseline_rmse": baseline})
    return pd.DataFrame(rows).sort_values("rmse_increase_mean", ascending=False)


def render_report(results: pd.DataFrame, manifest: dict, importance: pd.DataFrame) -> str:
    primary = results.loc[results.split.eq("temporal")].sort_values("rmse")
    selected = manifest["selected_candidate"]
    baseline = primary.loc[primary.candidate.eq("ridge_main")].iloc[0]
    champion = primary.loc[primary.candidate.eq(selected)].iloc[0]
    lines = ["# Stage 5: Feature Expansion and Model Optimisation", "", "## Validation and selection rules", "",
        "- Candidates are selected on the pooled RMSE of two earlier windows, 2024-07 to 2024-09 and 2024-10 to 2024-12.",
        "- 2025-01 to 2025-03 is the main validation window and is not used to select models, early-stopping rounds, blend weights or features. Earlier experiments have viewed it, so it is not an untouched final test set.",
        "- CatBoost and LightGBM hold out the last 3 months inside each outer training set for early stopping, then refit on the whole outer training set with the selected rounds.",
        "- The blend is fixed at 50% CatBoost / 50% LightGBM on the same features; weights are not tuned on the main validation.",
        "- All macro features are lagged by two months, and outer and inner validations freeze available source months at their own training cut-off. COE and stock values of future validation months are never used.", "",
        "## Main temporal validation", "", "| Candidate | RMSE | MAE | Mean prediction bias | Selected on earlier windows |", "| --- | ---: | ---: | ---: | --- |"]
    for row in primary.itertuples():
        lines.append(f"| {row.candidate} | {row.rmse:.4f} | {row.mae:.4f} | {row.bias:+.2f} | {'yes' if row.candidate == selected else ''} |")
    lines.extend(["", "## Interpretation", "",
        f"- The candidate selected on earlier windows is {selected}. Its main validation RMSE is {champion.rmse:.4f} against {baseline.rmse:.4f} for same-environment Ridge, an improvement of {(baseline.rmse - champion.rmse) / baseline.rmse:.2%}.",
        f"- Mean prediction bias changes from {baseline.bias:+.2f} for Ridge to {champion.bias:+.2f} for the candidate. Positive values mean overpredicted rent.",
        "- Model and feature selection details are in selection_backtests.csv. The lowest main-validation score need not be the candidate selected on earlier windows; the difference is kept to avoid post-hoc reselection.", "",
        "## Main feature groups", "", "| Feature group | RMSE increase after permutation |", "| --- | ---: |"])
    for row in importance.itertuples():
        lines.append(f"| {row.group} | {row.rmse_increase_mean:.4f} |")
    lines.extend(["", "Importance comes from grouped permutation on a fixed main-validation sample, repeated twice. It is for interpretation only and not used for feature selection. Correlated features can substitute for each other, so it is not a causal effect.", "",
        "## Temporal, macro and data limitations", "",
        "- The mrt_snapshot variant is exploratory only and excluded from recommended selection because historical opening dates are missing. Schools, malls and some block attributes are also static snapshots.",
        "- Monthly COE uses the mean premium across rounds and total bids / total quota. Stocks use equal-weight monthly returns of unadjusted CLOSE, mean daily volatility, share of rising securities and security coverage, without filling future prices.",
        "- The macro tables have no true publication dates. The two-month lag is a conservative availability assumption, not a no-leakage guarantee. Corporate actions such as splits may cause jumps in unadjusted prices.",
        "- Frozen macro values can be constant within a validation window, and future time fields can fall into the same tree leaves. Near-zero local permutation importance does not mean these variables had no role in training.",
        "- FEE and FURNISHED have no variation in the current data, so no fee or furnishing flags are added.",
        "- Quarterly validation does not guarantee performance over the long horizon to 2026-07; stage 6 needs longer-window evaluation and the final submission.",
        "- This round only tunes boosting rounds under preset parameters. If early stopping reaches the search cap, a higher-cap robustness check is needed; the parameters cannot be claimed globally optimal.",
        "- No test labels were read and no new competition submission was generated or uploaded. Saved models are diagnostic models at the main-validation cut-off, not final full-data models.", "",
        "## Outputs", "",
        "results.csv stores all outer experiments; group_errors.csv stores errors by town, flat type, flat model, floor area, flat age, month, region and address-match status.",
        "group_permutation_importance.csv holds candidate feature-group importance. native_feature_importance.csv holds tree training importance (CatBoost PredictionValuesChange, LightGBM gain); raw values are not comparable across algorithms.", "",
        "Model APIs follow the [CatBoost documentation](https://catboost.ai/docs/en/concepts/python-reference_catboostregressor) and the [LightGBM documentation](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMRegressor.html).", ""])
    return "\n".join(lines)


def run_stage5(output: Path, tree_iterations: int = 400, threads: int = 4, schemes: list[str] | None = None, workers: int = 3) -> dict:
    import catboost
    import lightgbm
    import joblib
    schemes = list(SCHEMES) if schemes is None else schemes
    if len(set(schemes)) != len(schemes) or not set(schemes).issubset(SCHEMES):
        raise ValueError("Schemes must be distinct known names")
    train_path = ROOT / "data/processed/train_location_features.csv"
    coe_path = ROOT / "data/interim/auxiliary/sg-coe-prices_clean.csv"
    stock_path = ROOT / "data/interim/auxiliary/sg-stock-prices_clean.csv"
    train = load_training(train_path)
    monthly, macro_audit = aggregate_macro(pd.read_csv(coe_path), pd.read_csv(stock_path))
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(output / "monthly_macro_features.csv")
    all_splits = validation_splits(train, 3, 2, True)
    by_name = {name: (fit, query) for name, fit, query in all_splits}
    ordered_splits = ["temporal_backtest_2", "temporal_backtest_1", "temporal"]
    records, predictions_by_split, primary_models, primary_frames = [], {}, {}, {}
    native_rows = []
    manifest = {
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__, "catboost": catboost.__version__, "lightgbm": lightgbm.__version__},
        "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in [train_path, coe_path, stock_path]},
        "tree_iterations_cap": tree_iterations, "threads": threads, "random_seed": 42,
        "workers": workers,
        "catboost": {"depth": 6, "learning_rate": 0.06, "l2_leaf_reg": 5.0},
        "lightgbm": {"num_leaves": 31, "learning_rate": 0.04, "min_child_samples": 40, "reg_lambda": 5.0, "colsample_bytree": 0.9},
        "early_stopping": {"inner_months": 3, "patience": 40, "outer_query_used": False},
        "macro": macro_audit, "feature_sets": {scheme: dict(zip(["categorical", "numeric"], feature_schema(scheme))) for scheme in schemes},
        "constant_fee_values": int(train.FEE.nunique()) if "FEE" in train else None,
        "constant_furnished_values": int(train.FURNISHED.nunique()) if "FURNISHED" in train else None,
        "selection_rule": "lowest pooled squared error on two earlier time windows; exclude MRT snapshot; equal blending weights fixed at 0.5",
    }
    manifest["implementation_sha256"] = implementation_hashes()
    signature = cache_identity(manifest)
    cache_dir = ROOT / "models/stage5_cache" / signature
    manifest["cache_signature"] = signature
    # A bounded search: compare algorithms on main/housing/amenities, and use
    # faster LightGBM for the full macro/snapshot sweep. Per-fit cache keys do
    # not depend on what other candidates happen to be included in the sweep.
    catboost_schemes = [scheme for scheme in schemes if scheme in {"main", "housing", "amenities"}]
    manifest["candidate_grid"] = {"catboost": catboost_schemes, "lightgbm": schemes, "blend": catboost_schemes}
    selected, selection_table = None, None
    for split_name in ordered_splits:
        fit, query = by_name[split_name]
        metadata = split_metadata(split_name, fit, query)
        predictions = {}
        candidates = [("ridge", "main"), ("ridge_spline", "main")]
        candidates += [("catboost", scheme) for scheme in catboost_schemes]
        candidates += [("lightgbm", scheme) for scheme in schemes]
        print(f"[{split_name}] evaluating {len(candidates)} candidates with {workers} workers", flush=True)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(cached_candidate, cache_dir, split_name, family, scheme, fit, query, monthly, tree_iterations, threads): (family, scheme) for family, scheme in candidates}
            for future in as_completed(futures):
                family, scheme = futures[future]
                candidate = f"{family}_{scheme}"
                bundle, prediction, details, query_frame = future.result()
                row = {**metadata, "candidate": candidate, "family": family, "scheme": scheme, **metrics(query[TARGET], prediction), **details}
                records.append(row)
                predictions[candidate] = prediction
                print(f"[{split_name}] {candidate}: RMSE {row['rmse']:.4f}; bias {row['bias']:+.2f}; iterations {details['iterations']}; {details['seconds']:.1f}s", flush=True)
                pd.DataFrame(records).to_csv(output / "results.csv", index=False)
                if split_name == "temporal":
                    primary_models[candidate] = bundle
                    primary_frames[candidate] = query_frame
                    if family in {"catboost", "lightgbm"}:
                        values = bundle.model.feature_importances_ if family == "catboost" else bundle.model.booster_.feature_importance(importance_type="gain")
                        for name, value in zip(bundle.features, values):
                            native_rows.append({"candidate": candidate, "feature": name, "importance": float(value), "method": "PredictionValuesChange" if family == "catboost" else "gain"})
        for scheme in catboost_schemes:
            candidate = f"blend_{scheme}"
            prediction = (predictions[f"catboost_{scheme}"] + predictions[f"lightgbm_{scheme}"]) / 2
            predictions[candidate] = prediction
            records.append({**metadata, "candidate": candidate, "family": "blend", "scheme": scheme, **metrics(query[TARGET], prediction), "iterations": None, "seconds": 0, "feature_count": len(feature_schema(scheme)[0] + feature_schema(scheme)[1])})
        predictions_by_split[split_name] = predictions
        pd.DataFrame({"source_row": query.index, "actual": query[TARGET].to_numpy(), **predictions}).to_csv(output / f"predictions_{split_name}.csv", index=False)
        pd.DataFrame(records).to_csv(output / "results.csv", index=False)
        if split_name == "temporal_backtest_1":
            selected, selection_table = select_candidate(pd.DataFrame(records))
            selection_table.to_csv(output / "selection_backtests.csv", index=False)
            print(f"LOCKED selection before primary evaluation: {selected}", flush=True)
            manifest["selected_candidate"] = selected
            (output / "selection_locked.json").write_text(json.dumps({"selected_candidate": selected, "selection_splits": ["temporal_backtest_2", "temporal_backtest_1"], "primary_used": False}, indent=2) + "\n", encoding="utf-8")
    results = pd.DataFrame(records)
    family, scheme = selected.split("_", 1)
    selected_bundles = [primary_models[f"catboost_{scheme}"], primary_models[f"lightgbm_{scheme}"]] if family == "blend" else [primary_models[selected]]
    primary = by_name["temporal"][1]
    selected_frame = primary_frames[f"catboost_{scheme}"] if family == "blend" else primary_frames[selected]
    manifest["selected_components"] = [f"catboost_{scheme}", f"lightgbm_{scheme}"] if family == "blend" else [selected]
    manifest["primary_best_observed"] = str(results.loc[results.split.eq("temporal")].sort_values("rmse").iloc[0].candidate)
    (output / "experiment_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    # Secondary random evaluation runs only locked components and the Ridge
    # baseline; it cannot influence feature/model selection.
    random_fit, random_query = by_name["random"]
    random_predictions = {}
    components = list(dict.fromkeys(["ridge_main", *manifest["selected_components"]]))
    for candidate in components:
        candidate_family, candidate_scheme = candidate.split("_", 1)
        if candidate == "ridge_spline_main":
            candidate_family, candidate_scheme = "ridge_spline", "main"
        print(f"[random] {candidate}", flush=True)
        _, prediction, details, _ = fit_candidate(candidate_family, candidate_scheme, random_fit, random_query, monthly, tree_iterations, threads)
        random_predictions[candidate] = prediction
        records.append({**split_metadata("random", random_fit, random_query), "candidate": candidate, "family": candidate_family, "scheme": candidate_scheme, **metrics(random_query[TARGET], prediction), **details})
    if family == "blend":
        prediction = np.mean([random_predictions[name] for name in manifest["selected_components"]], axis=0)
        records.append({**split_metadata("random", random_fit, random_query), "candidate": selected, "family": "blend", "scheme": scheme, **metrics(random_query[TARGET], prediction), "iterations": None, "seconds": 0, "feature_count": len(selected_bundles[0].features)})
    results = pd.DataFrame(records)
    results.to_csv(output / "results.csv", index=False)
    primary_predictions = predictions_by_split["temporal"]
    inspect_names = list(dict.fromkeys(["ridge_main", "ridge_spline_main", selected, manifest["primary_best_observed"]]))
    grouped_errors(primary, {name: primary_predictions[name] for name in inspect_names}).to_csv(output / "group_errors.csv", index=False)
    pd.DataFrame(native_rows).to_csv(output / "native_feature_importance.csv", index=False)
    importance = group_permutation(selected_bundles, selected_frame)
    importance.to_csv(output / "group_permutation_importance.csv", index=False)
    model_dir = ROOT / "models/stage5"
    model_dir.mkdir(parents=True, exist_ok=True)
    for name, bundle in zip(manifest["selected_components"], selected_bundles):
        # Built-in formats keep models portable and avoid storing __main__ classes.
        if bundle.family == "catboost":
            bundle.model.save_model(str(model_dir / f"{name}.cbm"))
        elif bundle.family == "lightgbm":
            bundle.model.booster_.save_model(str(model_dir / f"{name}.txt"))
        else:
            joblib.dump(bundle.model, model_dir / f"{name}.joblib")
        (model_dir / f"{name}.schema.json").write_text(json.dumps({"family": bundle.family, "cats": bundle.cats, "nums": bundle.nums, "levels": bundle.levels}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest["diagnostic_model_training_end"] = by_name["temporal"][0].RENT_APPROVAL_DATE.max().strftime("%Y-%m")
    manifest["splits"] = [split_metadata(name, *by_name[name]) for name in ordered_splits + ["random"]]
    (output / "experiment_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "stage5_report.md").write_text(render_report(results, manifest, importance), encoding="utf-8")
    print(f"Completed stage 5. Selected: {selected}. Report: {output / 'stage5_report.md'}", flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/experiments/stage5_v1")
    parser.add_argument("--tree-iterations", type=int, default=400)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--schemes", nargs="+", choices=SCHEMES, default=SCHEMES)
    args = parser.parse_args()
    if args.tree_iterations <= 0 or args.threads <= 0 or args.workers <= 0:
        parser.error("Iterations and thread count must be positive")
    run_stage5(args.output_dir, args.tree_iterations, args.threads, args.schemes, args.workers)


if __name__ == "__main__":
    main()
