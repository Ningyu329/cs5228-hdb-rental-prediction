"""Raw/clean, external resale and structural models under frozen-origin evaluation.

Never reads test rents or downloads public rental transactions. New long windows
compare matched standalone LightGBM controls, not an untrained production blend.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import json
import platform
import sys
import time
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if __package__ in {None, ""}:
    sys.path.insert(0, str(ROOT))

from src.features.build_raw_views import attach_raw_view, verify_alignment, RAW_CATS, RAW_NUMS
from src.features.build_resale_history import load_resale, causal_resale, FEATURES
from src.models.hierarchical_rent import HierarchicalRent
from src.models.run_stage5 import FittedModel, load_training, feature_schema, metrics
from src.models.run_rmse_improvement import dump, sha, signature, EARLIER, SPLITS
from src.models.run_tree_tuning import split_inner

OLD = ROOT / "reports/experiments/rmse_improvement_v1"
OLD_CACHE = ROOT / "models/rmse_improvement_cache/24cdf7b53bde332a047f"
ORIGINAL = {k: SPLITS[k] for k in [*EARLIER, "primary_2025", "long_12m"]}
LONG = {"origin_202212_16m": ("2023-01", "2024-04"),
        "origin_202306_16m": ("2023-07", "2024-10"),
        "origin_202311_16m": ("2023-12", "2025-03")}
CONTROL = "trend_damped24_weight25"
TIME_NUMS = {"RENT_YEAR", "RENT_MONTH", "RENT_MONTH_INDEX", "FLAT_AGE", "BUILDING_AGE", "RENT_MONTH_SIN", "RENT_MONTH_COS"}


def fit_model(fit, query, family, threads, rounds=None, valid=None, extra=False, structural=False):
    cats, nums = feature_schema("housing")
    if family == "catboost":
        cats, nums = cats + RAW_CATS, nums + RAW_NUMS
    if extra:
        nums += FEATURES
    if structural:
        cats = [c for c in cats if c != "FLAT_AGE_BAND"]
        nums = [c for c in nums if c not in TIME_NUMS]
    bundle = FittedModel(family, None, cats, nums,
                        {c: fit[c].fillna("missing").astype(str).unique().tolist() for c in cats})
    kwargs = {}
    if family == "catboost":
        from catboost import CatBoostRegressor
        model = CatBoostRegressor(iterations=2400 if rounds is None else rounds,
            depth=8, learning_rate=.05, l2_leaf_reg=10., loss_function="RMSE",
            random_seed=42, thread_count=threads, allow_writing_files=False, verbose=False)
        kwargs["cat_features"] = cats
        if valid is not None:
            kwargs.update(eval_set=(bundle.matrix(valid), valid.MONTHLY_RENT),
                          early_stopping_rounds=80, use_best_model=True)
    else:
        from lightgbm import LGBMRegressor, early_stopping
        model = LGBMRegressor(n_estimators=1000 if rounds is None else rounds,
            num_leaves=15, learning_rate=.04, min_child_samples=80, reg_lambda=10.,
            colsample_bytree=.9, max_depth=-1, objective="regression", random_state=42,
            n_jobs=threads, verbosity=-1, deterministic=True, force_col_wise=True)
        kwargs["categorical_feature"] = cats
        if valid is not None:
            kwargs.update(eval_set=[(bundle.matrix(valid), valid.MONTHLY_RENT)], eval_metric="rmse",
                          callbacks=[early_stopping(80, verbose=False)])
    model.fit(bundle.matrix(fit), fit.MONTHLY_RENT, **kwargs)
    bundle.model = model
    return bundle


def raw_predict(bundle, query):
    return np.asarray(bundle.model.predict(bundle.matrix(query)), float)


def choose(rows):
    names = sorted({r["candidate"] for r in rows if r["split"] in EARLIER})
    scores = []
    reference = {r["split"]: r["rmse"] for r in rows if r["candidate"] == "incumbent" and r["split"] in EARLIER}
    for name in names:
        group = [r for r in rows if r["candidate"] == name and r["split"] in EARLIER]
        if len(group) != 2 or {r["split"] for r in group} != set(EARLIER):
            raise ValueError("Selection needs both earlier quarters exactly once")
        pooled = float(np.sqrt(sum(r["rmse"] ** 2 * r["rows"] for r in group) / sum(r["rows"] for r in group)))
        scores.append(dict(candidate=name, pooled_rmse=pooled,
                           worst_quarter_delta=max(r["rmse"] - reference[r["split"]] for r in group)))
    baseline = next(r["pooled_rmse"] for r in scores if r["candidate"] == "incumbent")
    scores.sort(key=lambda r: (r["pooled_rmse"], r["candidate"]))
    for row in scores:
        row["gain"] = baseline - row["pooled_rmse"]
        row["eligible"] = row["gain"] >= 3 and row["worst_quarter_delta"] <= 3
    eligible = [r for r in scores if r["eligible"]]
    return scores, eligible[0]["candidate"] if eligible else "incumbent"


class Experiment:
    def __init__(self, output, threads):
        import catboost, lightgbm, sklearn, scipy
        self.output, self.threads = output, threads
        source = ROOT / "data/processed/train_location_features.csv"
        self.raw = pd.read_csv(ROOT / "data/raw/train.csv", dtype=str, keep_default_na=False)
        self.data = attach_raw_view(load_training(source), self.raw)
        self.resale_path = ROOT / "data/raw/external/hdb_20261019/resale.csv"
        self.resale = load_resale(self.resale_path)
        paths = [Path(__file__), ROOT / "src/features/build_raw_views.py", ROOT / "src/features/build_resale_history.py",
                 ROOT / "src/models/hierarchical_rent.py", ROOT / "src/models/run_stage5.py",
                 ROOT / "src/models/run_rmse_improvement.py", ROOT / "src/models/run_tree_tuning.py",
                 ROOT / "src/features/build_stage5_features.py", ROOT / "src/models/train_baseline.py",
                 ROOT / "src/data/clean_data.py", ROOT / "src/features/build_temporal_features.py"]
        refs = [OLD / "selection_locked.json", *[OLD / f"predictions_{s}.csv" for s in ORIGINAL],
                *[OLD_CACHE / f"pred_{s}_catboost_cap1600_w0_h0_hist0_direct.npz" for s in ORIGINAL]]
        self.manifest = dict(
            input_sha256={str(p.relative_to(ROOT)): sha(p) for p in [source, self.resale_path, ROOT / "data/raw/train.csv"]},
            implementation_sha256={str(p.relative_to(ROOT)): sha(p) for p in paths},
            reference_sha256={str(p.relative_to(ROOT)): sha(p) for p in refs},
            environment=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                             catboost=catboost.__version__, lightgbm=lightgbm.__version__, sklearn=sklearn.__version__, scipy=scipy.__version__),
            splits=dict(ORIGINAL, **LONG), threads=threads, seed=42,
            recipes=["raw_cat", "resale_lgb", "hierarchy", "hierarchy_residual_lgb"],
            raw="Normalized keys retained, plus raw street/type and 2 formatting indicators; same CatBoost 2400 trajectory, <=1600 best prefix",
            resale="24-month snapshot; 2 calendar month lag; query frozen at fit end minus lag; latest-file historical revisions unverified",
            hierarchy=dict(ridge_alpha=30, future_damping_months=6, recent_group_slope_scale=.25),
            blends=[.25, 1.], selection="Earlier two quarters only; >=3 pooled gain and no quarter loss >3; no automatic production replacement",
            long_window_anchor="Matched housing LightGBM ONLY, not incumbent production blend",
            test_inputs="Audit dates and alignment only; no test fields used by model fitting or selection",
            public_rental_data_downloaded=False, property_features_used=False, submission_generated=False,
            limitations=["Repeatedly inspected and overlapping historical windows, not pristine holdouts",
                         "Frozen event lag cannot establish publication vintage or undo historical revisions",
                         "Latest property snapshot excluded; no individual unit identifiers"])
        self.run_id = signature(self.manifest)
        path = output / "manifest.json"
        if path.exists() and json.loads(path.read_text())["run_id"] != self.run_id:
            raise ValueError("Changed code/inputs: use a fresh output directory")
        dump(path, dict(self.manifest, run_id=self.run_id))
        self.cache = ROOT / "models/data_rethink_cache" / self.run_id
        self.cache.mkdir(parents=True, exist_ok=True)
        self.rows, self.details, self.monthly = [], [], []

    def audit(self):
        audits = {}
        for split in ["train", "test"]:
            raw = pd.read_csv(ROOT / f"data/raw/{split}.csv", dtype=str, keep_default_na=False)
            clean = pd.read_csv(ROOT / f"data/interim/{split}_clean.csv", dtype=str, keep_default_na=False)
            audits[split] = dict(rows=len(raw), fields=verify_alignment(raw, clean))
        keys = ["TOWN", "BLOCK", "STREET", "FLAT_TYPE"]
        grouped = self.data.groupby(keys)[["FLOOR_AREA_SQM", "FLAT_MODEL", "LEASE_COMMENCE_DATE"]].nunique(dropna=False)
        audits["granularity"] = dict(block_type_groups=len(grouped),
            groups_with_varying_attributes={c: int(grouped[c].gt(1).sum()) for c in grouped},
            interpretation="Static block+type attributes, not evidence of exact unit measurements; 2026 origin not verified")
        test = pd.read_csv(ROOT / "data/raw/test.csv", usecols=["RENT_APPROVAL_DATE"])
        audits["test_dates"] = test.RENT_APPROVAL_DATE.value_counts().sort_index().to_dict()
        audits["resale"] = dict(rows=len(self.resale), min_month=str(self.resale.RENT_APPROVAL_DATE.min().date()),
            max_month=str(self.resale.RENT_APPROVAL_DATE.max().date()), receipt=json.loads(self.resale_path.with_name("resale_receipt.json").read_text()))
        audits["rule_confirmation"] = "User states course allows additional public data, 2026-10-19"
        dump(self.output / "data_audit.json", audits)

    def frames(self, split):
        start, end = dict(ORIGINAL, **LONG)[split]
        m = self.data.RENT_APPROVAL_DATE.dt.to_period("M")
        return self.data.loc[m.lt(pd.Period(start))].copy(), self.data.loc[m.between(pd.Period(start), pd.Period(end))].copy()

    def prepare(self, fit, query, recipe):
        a, b = fit.copy(), query.drop(columns="MONTHLY_RENT", errors="ignore").copy()
        structure = None
        if recipe == "resale_lgb":
            key = signature(dict(origin=str(fit.RENT_APPROVAL_DATE.max()), rows=len(fit), raw=sha(self.resale_path),
                                 query_rows=signature(dict(indices=query.index.tolist())),
                                 code=sha(ROOT / "src/features/build_resale_history.py")))
            path = self.cache / f"features_{key}.joblib"
            if path.exists():
                a, saved = joblib.load(path)
                if not saved.index.equals(b.index):
                    raise ValueError("Cached resale query order mismatch")
                b = saved
            else:
                a, b = causal_resale(a, b, self.resale)
                joblib.dump((a, b), path)
        elif recipe == "hierarchy_residual_lgb":
            structure = HierarchicalRent().fit(a)
            a["MONTHLY_RENT"] = a.MONTHLY_RENT.to_numpy() - structure.predict(a)
        return a, b, structure

    def component(self, split, recipe):
        fit, query = self.frames(split)
        directory = self.cache / split / recipe
        result = directory / "result.json"
        if result.exists():
            details = json.loads(result.read_text())
            with np.load(directory / "prediction.npz") as z:
                np.testing.assert_array_equal(z["source_row"], query.index)
                prediction = z["prediction"].copy()
        else:
            print(f"START {split} {recipe}", flush=True)
            started = time.perf_counter()
            directory.mkdir(parents=True, exist_ok=True)
            if recipe == "hierarchy":
                structure = HierarchicalRent().fit(fit)
                prediction = np.maximum(structure.predict(query), 0)
                bundle, rounds, inner_metric = None, None, None
            else:
                inner_fit, inner_valid = split_inner(fit)
                a, b, _ = self.prepare(inner_fit, inner_valid, recipe)
                inner_structure = HierarchicalRent().fit(inner_fit) if recipe == "hierarchy_residual_lgb" else None
                labels = inner_valid.MONTHLY_RENT.to_numpy(dtype=float).copy()
                if inner_structure is not None:
                    labels -= inner_structure.predict(inner_valid)
                b["MONTHLY_RENT"] = labels
                family = "catboost" if recipe == "raw_cat" else "lightgbm"
                kwargs = dict(extra=recipe == "resale_lgb", structural=inner_structure is not None)
                tuned = fit_model(a, b, family, self.threads, valid=b, **kwargs)
                if family == "catboost":
                    history = tuned.model.get_evals_result()["validation"]["RMSE"][:1600]
                    rounds = int(np.argmin(history)) + 1
                    # Restrict the model to exactly the same prefix used for refit.
                    tuned.model.shrink(rounds)
                else:
                    rounds = max(1, int(tuned.model.best_iteration_))
                p = raw_predict(tuned, b)
                if inner_structure is not None:
                    p += inner_structure.predict(inner_valid)
                inner_metric = metrics(inner_valid.MONTHLY_RENT, np.maximum(p, 0))
                del tuned
                a, b, structure = self.prepare(fit, query, recipe)
                bundle = fit_model(a, b, family, self.threads, rounds=rounds, **kwargs)
                prediction = raw_predict(bundle, b)
                if structure is not None:
                    prediction += structure.predict(query)
                prediction = np.maximum(prediction, 0)
            details = dict(split=split, recipe=recipe, iterations=rounds, inner=inner_metric,
                fit_end=str(fit.RENT_APPROVAL_DATE.max().date()), rows=len(query),
                metrics=metrics(query.MONTHLY_RENT, prediction), seconds=time.perf_counter() - started)
            if recipe == "resale_lgb":
                details["coverage"] = {level: float((b[f"RESALE_{level}_COUNT"] > 0).mean()) for level in ["BLOCK", "STREET", "TOWN"]}
                details["resale_max_event_month"] = str(fit.RENT_APPROVAL_DATE.max().to_period("M") - 2)
            payload = dict(bundle=bundle, structure=structure, details=details, recipe=recipe)
            joblib.dump(payload, directory / "model.joblib")
            np.savez_compressed(directory / "prediction.npz", source_row=query.index, prediction=prediction)
            # Independently reload the saved model before recording completion.
            loaded = joblib.load(directory / "model.joblib")
            if recipe == "hierarchy":
                reloaded = np.maximum(loaded["structure"].predict(query), 0)
            else:
                reloaded = raw_predict(loaded["bundle"], b)
                if loaded["structure"] is not None:
                    reloaded += loaded["structure"].predict(query)
                reloaded = np.maximum(reloaded, 0)
            np.testing.assert_allclose(reloaded, prediction, rtol=1e-10, atol=1e-8)
            dump(result, details)
        if not np.isfinite(prediction).all() or len(prediction) != len(query):
            raise ValueError("Invalid predictions")
        self.details.append(details)
        print(f"DONE {split} {recipe} RMSE={details['metrics']['rmse']:.4f}", flush=True)
        return prediction

    def record(self, split, candidate, pred, anchor):
        fit, query = self.frames(split)
        row = dict(split=split, candidate=candidate, anchor=anchor, rows=len(query), **metrics(query.MONTHLY_RENT, pred))
        self.rows.append(row)
        month = query.RENT_APPROVAL_DATE.dt.to_period("M")
        origin = fit.RENT_APPROVAL_DATE.max().to_period("M")
        for m in sorted(month.unique()):
            mask = month.eq(m).to_numpy()
            self.monthly.append(dict(split=split, candidate=candidate, month=str(m), horizon=m.ordinal - origin.ordinal,
                                     rows=int(mask.sum()), **metrics(query.loc[mask, "MONTHLY_RENT"], pred[mask])))
        np.savez_compressed(self.cache / f"candidate_{split}_{candidate}.npz", source_row=query.index, prediction=pred)

    def evaluate(self, split, long=False):
        _, query = self.frames(split)
        if not long:
            old = pd.read_csv(OLD / f"predictions_{split}.csv")
            np.testing.assert_array_equal(old.source_row, query.index)
            np.testing.assert_array_equal(old.actual, query.MONTHLY_RENT)
            anchor = old[CONTROL].to_numpy()
            self.record(split, "incumbent", anchor, "locked_production")
            # A replacement, not an extra full-model blend: only its 37.5% CatBoost changes.
            raw = self.component(split, "raw_cat")
            with np.load(OLD_CACHE / f"pred_{split}_catboost_cap1600_w0_h0_hist0_direct.npz") as z:
                np.testing.assert_array_equal(z["source_row"], query.index)
                normalized = z["prediction"].copy()
            self.record(split, "raw_cat_only", raw, "normalized_cat_only")
            self.record(split, "normalized_cat_only", normalized, "normalized_cat_only")
            self.record(split, "raw_cat_replacement", anchor + .375 * (raw - normalized), "locked_production")
        else:
            anchor = self.component(split, "housing_lgb")
            self.record(split, "housing_lgb", anchor, "housing_lgb")
        for recipe in ["resale_lgb", "hierarchy", "hierarchy_residual_lgb"]:
            pred = self.component(split, recipe)
            self.record(split, recipe + "_only", pred, "housing_lgb" if long else "locked_production")
            self.record(split, recipe + "_blend25", .75 * anchor + .25 * pred, "housing_lgb" if long else "locked_production")
        dump(self.output / "results.json", self.rows)
        dump(self.output / "fit_details.json", self.details)
        dump(self.output / "monthly_errors.json", self.monthly)

    def run(self):
        self.audit()
        dump(self.output / "validation_plan.json", dict(splits=dict(ORIGINAL, **LONG),
            selection_splits=EARLIER, diagnostic_only=["primary_2025", "long_12m", *LONG],
            test_horizon="380 rows in existing March 2025; remaining 49,620 rows April 2025–July 2026 (1–16 months)",
            long_anchor="Housing LightGBM; do not label as production model", future_labels_available_to_training=False))
        for split in EARLIER:
            self.evaluate(split)
        # Component-only controls are useful comparisons, but are not eligible to replace the blend.
        eligible = [r for r in self.rows if r["candidate"] not in {"raw_cat_only", "normalized_cat_only"}]
        ranking, selected = choose(eligible)
        dump(self.output / "selection_locked.json", dict(selected=selected, ranking=ranking,
            primary_used_for_selection=False, production_replaced=False))
        for split in ["primary_2025", "long_12m"]:
            self.evaluate(split)
        for split in LONG:
            self.evaluate(split, long=True)
        dump(self.output / "completion.json", dict(run_id=self.run_id, complete=True, scores=len(self.rows),
            model_records=len(self.details), selected=selected, submission_generated=False, production_replaced=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/experiments/data_rethink_v3")
    parser.add_argument("--threads", type=int, default=3)
    args = parser.parse_args()
    Experiment(args.output_dir.resolve(), args.threads).run()
