"""XGBoost comparison with nested time splits, native categories and one-hot."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import platform
import sys
import time

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[2]
if __package__ in {None, ""}:
    sys.path.insert(0, str(ROOT))

from src.features.build_neighborhood_features import ALL_FEATURES, area_values, causal_features
from src.features.build_temporal_features import fit_trend, trend_values
from src.models.run_stage5 import feature_schema, load_training, metrics
from src.models.run_rmse_improvement import EARLIER, SPLITS, dump, sha, signature
from src.models.run_tree_tuning import split_inner
from src.models.train_baseline import TARGET

REFERENCE = ROOT / "reports/experiments/rmse_improvement_v1"
CONTROL = "trend_damped24_weight25"
TIME_NUMS = {"RENT_YEAR", "RENT_MONTH", "RENT_MONTH_INDEX", "FLAT_AGE", "BUILDING_AGE", "RENT_MONTH_SIN", "RENT_MONTH_COS"}


def grid():
    base = dict(encoding="native", target="direct", history=False, cap=2000,
                depth=6, child=40., l2=20., rate=.04)
    return {
        "native_d4": dict(base, depth=4, child=80., rate=.05),
        "native_d6": dict(base),
        "native_d8": dict(base, depth=8, child=10., l2=10.),
        "native_d10": dict(base, depth=10, child=5., l2=5.),
        "onehot_d6": dict(base, encoding="onehot"),
        "onehot_d8": dict(base, encoding="onehot", depth=8, child=10., l2=10.),
        "native_unit_d6": dict(base, target="unit"),
        "native_history_d6": dict(base, history=True),
        "native_trend_d6": dict(base, target="trend"),
    }


@dataclass
class Bundle:
    model: object
    cats: list[str]
    nums: list[str]
    levels: dict
    encoding: str
    target: str
    trend: dict | None = None
    encoder: object = None

    def matrix(self, frame):
        values = frame[self.cats].fillna("missing").astype(str)
        if self.encoding == "native":
            out = frame[self.cats + self.nums].copy()
            for column in self.cats:
                known = values[column].where(values[column].isin(self.levels[column]), pd.NA)
                out[column] = pd.Categorical(known, categories=self.levels[column])
            return out
        numeric = frame[self.nums].to_numpy(dtype=float)
        n, width = numeric.shape
        # Explicitly store numeric zeroes: sparse absence means missing in XGB,
        # whereas a measured zero must remain a real value.
        numbers = sparse.csr_matrix((numeric.ravel(), np.tile(np.arange(width), n),
            np.arange(n+1)*width), shape=(n, width))
        return sparse.hstack([self.encoder.transform(values), numbers], format="csr")

    def response(self, frame):
        y = frame[TARGET].to_numpy(dtype=float)
        if self.target == "unit":
            areas = area_values(frame)
            weights = areas**2
            return y/areas, weights/weights.mean()
        if self.target == "trend":
            y = y - trend_values(self.trend, frame)
        return y, None

    def predict(self, frame):
        pred = np.asarray(self.model.predict(self.matrix(frame)), dtype=float)
        if self.target == "unit":
            pred *= area_values(frame)
        if self.target == "trend":
            pred += trend_values(self.trend, frame)
        return np.maximum(pred, 0.)

    def save(self, directory):
        directory.mkdir(parents=True, exist_ok=True)
        self.model.save_model(directory / "model.json")
        dump(directory / "schema.json", dict(cats=self.cats, nums=self.nums, levels=self.levels,
            encoding=self.encoding, target=self.target, trend=self.trend))
        if self.encoder is not None:
            joblib.dump(self.encoder, directory / "encoder.joblib")


def load_bundle(directory):
    schema = json.loads((directory / "schema.json").read_text())
    if schema["trend"] is not None:
        schema["trend"]["monthly_levels"] = {int(k): v for k, v in schema["trend"]["monthly_levels"].items()}
    model = xgb.XGBRegressor()
    model.load_model(directory / "model.json")
    encoder = joblib.load(directory / "encoder.joblib") if schema["encoding"] == "onehot" else None
    return Bundle(model=model, encoder=encoder, **schema)


def prepare(fit, query, spec, cache=None):
    cats, nums = feature_schema("housing")
    left, right = fit.copy(), query.drop(columns=TARGET, errors="ignore").copy()
    if spec["history"]:
        left, right = causal_features(left, right, cache=cache)
        nums += ALL_FEATURES
    trend = None
    if spec["target"] == "trend":
        trend = fit_trend(fit, "damped24")
        cats = [c for c in cats if c != "FLAT_AGE_BAND"]
        nums = [c for c in nums if c not in TIME_NUMS]
    return left, right, cats, nums, trend


def train(spec, fit, threads=3, rounds=None, valid=None, cache=None):
    query = fit.iloc[:0] if valid is None else valid
    left, right, cats, nums, trend = prepare(fit, query, spec, cache)
    levels = {c: left[c].fillna("missing").astype(str).unique().tolist() for c in cats}
    encoder = None
    if spec["encoding"] == "onehot":
        encoder = OneHotEncoder(handle_unknown="ignore", min_frequency=2, sparse_output=True)
        encoder.fit(left[cats].fillna("missing").astype(str))
    bundle = Bundle(None, cats, nums, levels, spec["encoding"], spec["target"], trend, encoder)
    model = xgb.XGBRegressor(n_estimators=spec["cap"] if rounds is None else rounds,
        objective="reg:squarederror", eval_metric="rmse", tree_method="hist", device="cpu",
        enable_categorical=spec["encoding"] == "native", max_depth=spec["depth"],
        min_child_weight=spec["child"], reg_lambda=spec["l2"], learning_rate=spec["rate"],
        subsample=.9, colsample_bytree=.9, max_bin=256, max_cat_to_onehot=4,
        max_cat_threshold=64, max_cached_hist_node=256, random_state=42, n_jobs=threads,
        early_stopping_rounds=80 if valid is not None else None)
    y, weights = bundle.response(fit)
    kwargs = dict(sample_weight=weights, verbose=False)
    if valid is not None:
        vy, vw = bundle.response(valid)
        kwargs["eval_set"] = [(bundle.matrix(right), vy)]
        if vw is not None:
            kwargs["sample_weight_eval_set"] = [vw]
    model.fit(bundle.matrix(left), y, **kwargs)
    bundle.model = model
    return bundle


def ranking(rows):
    table = pd.DataFrame(rows)
    table = table.loc[table.split.isin(EARLIER)].copy()
    if table.duplicated(["split", "candidate"]).any():
        raise ValueError("Duplicate selection rows")
    for _, group in table.groupby("candidate"):
        if set(group.split) != set(EARLIER):
            raise ValueError("Incomplete earlier quarters")
    table["sse"] = table.rmse**2*table.rows
    scores = table.groupby("candidate").agg(sse=("sse", "sum"), rows=("rows", "sum")).reset_index()
    scores["pooled_rmse"] = np.sqrt(scores.sse/scores.rows)
    scores["below_300"] = scores.pooled_rmse < 300
    scores["below_200"] = scores.pooled_rmse < 200
    scores = scores.sort_values(["pooled_rmse", "candidate"]).reset_index(drop=True)
    best_pure = str(scores.loc[scores.candidate.str.endswith("_blend100")].iloc[0].candidate)
    best_new = str(scores.loc[~scores.candidate.eq("incumbent")].iloc[0].candidate)
    return scores, best_pure, best_new


class Experiment:
    def __init__(self, output, threads, include_onehot=False):
        import scipy, sklearn
        self.output, self.threads = output.resolve(), threads
        self.output.mkdir(parents=True, exist_ok=True)
        self.specs = {name: spec for name, spec in grid().items()
                      if include_onehot or spec["encoding"] != "onehot"}
        source = ROOT / "data/processed/train_location_features.csv"
        code = [Path(__file__), ROOT / "src/features/build_neighborhood_features.py",
            ROOT / "src/features/build_temporal_features.py", ROOT / "src/models/run_stage5.py",
            ROOT / "src/features/build_stage5_features.py", ROOT / "src/models/train_baseline.py",
            ROOT / "src/models/run_tree_tuning.py", ROOT / "src/models/run_rmse_improvement.py",
            ROOT / "src/models/run_location_ablation.py", ROOT / "requirements-xgboost.txt"]
        self.manifest = dict(source=str(source.relative_to(ROOT)), source_sha256=sha(source),
            implementation_sha256={str(p.relative_to(ROOT)): sha(p) for p in code},
            reference_sha256={str(p.relative_to(ROOT)): sha(p) for p in
                [REFERENCE/"selection_locked.json", *[REFERENCE/f"predictions_{s}.csv" for s in SPLITS]]},
            environment=dict(python=platform.python_version(), xgboost=xgb.__version__,
                pandas=pd.__version__, numpy=np.__version__, sklearn=sklearn.__version__, scipy=scipy.__version__),
            grid=self.specs, include_onehot=include_onehot, threads=threads, seed=42,
            early_stopping="inner last 3 months only, patience 80",
            selection="pooled earlier quarters; lock best pure XGB and best new mixture before diagnostics",
            blend_weights=[.25, .5, 1.], below_300_is_200s_goal=True, below_200_is_leaderboard_aspiration=True,
            production_replaced=False, primary_used_for_selection=False, previously_inspected_windows=True,
            test_data_read=False, submission_generated=False,
            diagnostics=["primary_2025", "long_6m", "long_12m"])
        self.run_id = signature(self.manifest)
        path = self.output/"experiment_manifest.json"
        if path.exists() and json.loads(path.read_text())["run_id"] != self.run_id:
            raise ValueError("Changed experiment; choose a new output directory")
        dump(path, dict(self.manifest, run_id=self.run_id))
        self.cache = ROOT/"models/xgboost_cache"/self.run_id
        self.feature_cache = self.cache/"history_features"
        self.data = load_training(source)
        self.rows, self.predictions, self.details = [], {}, []

    def frames(self, split):
        start, end = [pd.Period(s) for s in SPLITS[split]]
        months = self.data.RENT_APPROVAL_DATE.dt.to_period("M")
        return self.data.loc[months.lt(start)].copy(), self.data.loc[months.between(start, end)].copy()

    def component(self, split, name):
        key = (split, name)
        if key in self.predictions:
            return self.predictions[key]
        fit, query = self.frames(split)
        spec = self.specs[name]
        directory = self.cache/split/name
        if (directory/"result.json").exists():
            detail = json.loads((directory/"result.json").read_text())
            with np.load(directory/"prediction.npz") as saved:
                np.testing.assert_array_equal(saved["source_row"], query.index)
                pred = saved["prediction"].copy()
        else:
            print(f"START {split} {name}", flush=True)
            start = time.perf_counter()
            inner_fit, inner_valid = split_inner(fit)
            tuned = train(spec, inner_fit, self.threads, valid=inner_valid, cache=self.feature_cache)
            rounds = tuned.model.best_iteration+1
            inner_metrics = metrics(inner_valid[TARGET], tuned.predict(prepare(inner_fit, inner_valid, spec, self.feature_cache)[1]))
            del tuned
            bundle = train(spec, fit, self.threads, rounds=rounds, cache=self.feature_cache)
            xfit, xquery, _, _, _ = prepare(fit, query, spec, self.feature_cache)
            pred = bundle.predict(xquery)
            detail = dict(split=split, component=name, iterations=rounds, hit_cap=rounds == spec["cap"],
                fit_rows=len(fit), query_rows=len(query), fit_end=str(fit.RENT_APPROVAL_DATE.max().date()),
                inner_fit_end=str(inner_fit.RENT_APPROVAL_DATE.max().date()),
                inner_valid_start=str(inner_valid.RENT_APPROVAL_DATE.min().date()),
                inner_valid_end=str(inner_valid.RENT_APPROVAL_DATE.max().date()),
                inner=inner_metrics, training=metrics(fit[TARGET], bundle.predict(xfit)), outer=metrics(query[TARGET], pred),
                seconds=time.perf_counter()-start)
            bundle.save(directory)
            np.savez_compressed(directory/"prediction.npz", source_row=query.index, prediction=pred)
            dump(directory/"result.json", detail)
        if len(pred) != len(query) or not np.isfinite(pred).all():
            raise ValueError("Invalid predictions")
        self.predictions[key] = pred
        self.details.append(detail)
        print(f"DONE {split} {name} RMSE={detail['outer']['rmse']:.4f} rounds={detail['iterations']}", flush=True)
        return pred

    def evaluate(self, split, name):
        if (split, name) in self.predictions:
            return
        _, query = self.frames(split)
        old = pd.read_csv(REFERENCE/f"predictions_{split}.csv")
        np.testing.assert_array_equal(old.source_row, query.index)
        np.testing.assert_array_equal(old.actual, query[TARGET])
        pred = old[CONTROL].to_numpy()
        if name != "incumbent":
            component, percent = name.rsplit("_blend", 1)
            weight = int(percent)/100
            pred = (1-weight)*pred + weight*self.component(split, component)
        self.predictions[split, name] = pred
        self.rows.append(dict(split=split, candidate=name, rows=len(query), **metrics(query[TARGET], pred)))

    def save(self):
        pd.DataFrame(self.rows).to_csv(self.output/"results.csv", index=False)
        dump(self.output/"fit_details.json", self.details)
        for split in {s for s, _ in self.predictions}:
            _, query = self.frames(split)
            table = pd.DataFrame(dict(source_row=query.index, actual=query[TARGET].to_numpy()))
            for (s, name), pred in self.predictions.items():
                if s == split:
                    table[name] = pred
            table.to_csv(self.output/f"predictions_{split}.csv", index=False)

    def run(self):
        for split in EARLIER:
            self.evaluate(split, "incumbent")
            for component in self.specs:
                for percent in [25, 50, 100]:
                    self.evaluate(split, f"{component}_blend{percent}")
                self.save()
        scores, pure, mixture = ranking(self.rows)
        scores.to_csv(self.output/"selection.csv", index=False)
        lock = dict(run_id=self.run_id, best_pure=pure, best_new=mixture,
                    selection_splits=EARLIER, primary_used=False, production_replaced=False)
        path = self.output/"selection_locked.json"
        if path.exists() and json.loads(path.read_text()) != lock:
            raise ValueError("Selection lock changed")
        dump(path, lock)
        print(f"LOCKED pure={pure} best_new={mixture}", flush=True)
        for split in self.manifest["diagnostics"]:
            for name in dict.fromkeys(["incumbent", pure, mixture]):
                self.evaluate(split, name)
            self.save()
        dump(self.output/"completion.json", dict(completed=True, **lock))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT/"reports/experiments/xgboost_native_v1")
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--include-onehot", action="store_true", help="Include two costly high-cardinality one-hot settings; use a fresh output directory")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    Experiment(args.output_dir, args.threads, args.include_onehot).run()
