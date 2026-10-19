"""Bounded causal distribution/spatial/unit-rent experiments; no test access."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys
import time

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if __package__ in {None, ""}:
    sys.path.insert(0, str(ROOT))

from src.features.build_neighborhood_features import (
    ALL_FEATURES, DISTRIBUTION, SPATIAL, area_values, causal_features,
)
from src.models.run_stage5 import FittedModel, feature_schema, load_training, metrics
from src.models.run_rmse_improvement import SPLITS, EARLIER, dump, sha, signature
from src.models.run_tree_tuning import split_inner, chosen_rounds
from src.models.train_baseline import TARGET

CONTROL = "trend_damped24_weight25"
OLD = ROOT / "reports/experiments/rmse_improvement_v1"
OLD_CACHE = ROOT / "models/rmse_improvement_cache/24cdf7b53bde332a047f"


def grid():
    base = dict(family="lightgbm", cap=1000, patience=80,
        params=dict(num_leaves=15, learning_rate=.04, min_child_samples=80,
                    reg_lambda=10., colsample_bytree=.9))
    return {
        "housing_control": dict(base, features=[], unit=False),
        "unit_area": dict(base, features=[], unit=True),
        "distribution": dict(base, features=DISTRIBUTION, unit=False),
        "spatial": dict(base, features=SPATIAL, unit=False),
        "distribution_spatial": dict(base, features=ALL_FEATURES, unit=False),
        "unit_area_all": dict(base, features=ALL_FEATURES, unit=True),
        "flexible_all": dict(base, cap=1500, features=ALL_FEATURES, unit=False,
            params=dict(num_leaves=31, learning_rate=.04, min_child_samples=40,
                        reg_lambda=5., colsample_bytree=.9)),
        "cat_shallow": dict(family="catboost", cap=4000, patience=100,
            params=dict(depth=5, learning_rate=.05, l2_leaf_reg=50.), features=[], unit=False),
    }


def response(frame, unit):
    y = frame[TARGET].to_numpy(dtype=float)
    if not unit:
        return y, None
    areas = area_values(frame)
    weights = areas ** 2
    return y / areas, weights / weights.mean()


def predict(bundle, query, unit=False):
    prediction = np.asarray(bundle.model.predict(bundle.matrix(query)), dtype=float)
    if unit:
        prediction *= area_values(query)
    return np.maximum(prediction, 0.)


def train(spec, fit, threads, rounds=None, valid=None):
    cats, nums = feature_schema("housing")
    levels = {c: fit[c].fillna("missing").astype(str).unique().tolist() for c in cats}
    bundle = FittedModel(spec["family"], None, cats, nums + spec["features"], levels)
    y, weights = response(fit, spec["unit"])
    count = spec["cap"] if rounds is None else rounds
    kwargs = dict(sample_weight=weights)
    if spec["family"] == "lightgbm":
        from lightgbm import LGBMRegressor, early_stopping
        model = LGBMRegressor(n_estimators=count, **spec["params"], max_depth=-1,
            objective="regression", random_state=42, n_jobs=threads, verbosity=-1,
            deterministic=True, force_col_wise=True)
        kwargs["categorical_feature"] = cats
        if valid is not None:
            vy, vw = response(valid, spec["unit"])
            kwargs.update(eval_X=bundle.matrix(valid), eval_y=vy, eval_metric="rmse",
                callbacks=[early_stopping(spec["patience"], verbose=False)])
            if vw is not None:
                kwargs["eval_sample_weight"] = [vw]
    else:
        from catboost import CatBoostRegressor
        model = CatBoostRegressor(iterations=count, **spec["params"], loss_function="RMSE",
            random_seed=42, thread_count=threads, allow_writing_files=False, verbose=False)
        kwargs["cat_features"] = cats
        if valid is not None:
            kwargs.update(eval_set=(bundle.matrix(valid), valid[TARGET]),
                early_stopping_rounds=spec["patience"], use_best_model=True)
    model.fit(bundle.matrix(fit), y, **kwargs)
    bundle.model = model
    return bundle


def choose(rows):
    """Only earlier quarters select; minimum 1 RMSE gain, no >1 quarter loss."""
    earlier = pd.DataFrame(rows)
    earlier = earlier.loc[earlier.split.isin(EARLIER)].copy()
    if earlier.duplicated(["split", "candidate"]).any():
        raise ValueError("Duplicate selection entries")
    for _, group in earlier.groupby("candidate"):
        if set(group.split) != set(EARLIER):
            raise ValueError("Every candidate needs both quarters")
    earlier["sse"] = earlier.rmse ** 2 * earlier.rows
    table = earlier.groupby("candidate").agg(sse=("sse", "sum"), rows=("rows", "sum")).reset_index()
    table["pooled_rmse"] = np.sqrt(table.sse / table.rows)
    reference = earlier.loc[earlier.candidate.eq("incumbent")].set_index("split").rmse
    base = float(table.loc[table.candidate.eq("incumbent"), "pooled_rmse"].iloc[0])
    deltas = {name: float((group.set_index("split").rmse - reference).max())
              for name, group in earlier.groupby("candidate")}
    table["worst_quarter_delta"] = table.candidate.map(deltas)
    table["gain"] = base - table.pooled_rmse
    table["eligible"] = (table.gain >= 1.) & (table.worst_quarter_delta <= 1.)
    table = table.sort_values(["pooled_rmse", "candidate"]).reset_index(drop=True)
    challenger = str(table.loc[~table.candidate.eq("incumbent")].iloc[0].candidate)
    accepted = table.loc[table.eligible]
    selected = str(accepted.iloc[0].candidate) if len(accepted) else "incumbent"
    return table, selected, challenger


class Experiment:
    def __init__(self, output, threads):
        import catboost, lightgbm, sklearn
        self.output, self.threads = output.resolve(), threads
        self.output.mkdir(parents=True, exist_ok=True)
        self.specs = grid()
        source = ROOT / "data/processed/train_location_features.csv"
        paths = [Path(__file__), ROOT / "src/features/build_neighborhood_features.py",
            ROOT / "src/models/run_stage5.py", ROOT / "src/models/run_rmse_improvement.py",
            ROOT / "src/models/run_tree_tuning.py", ROOT / "src/models/train_baseline.py",
            ROOT / "src/models/run_location_ablation.py", ROOT / "src/features/build_stage5_features.py",
            ROOT / "src/features/build_temporal_features.py"]
        self.manifest = dict(source=str(source.relative_to(ROOT)), source_sha256=sha(source),
            implementation_sha256={str(p.relative_to(ROOT)): sha(p) for p in paths},
            reference_sha256={str(p.relative_to(ROOT)): sha(p) for p in
                [OLD / "selection_locked.json", *[OLD / f"predictions_{s}.csv" for s in [*EARLIER, "primary_2025", "long_6m"]]]},
            environment=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                sklearn=sklearn.__version__, catboost=catboost.__version__, lightgbm=lightgbm.__version__),
            grid=self.specs, threads=threads, seed=42, history_months=12, spatial_k=[16, 64],
            spatial_reference="one mean unit rent per block+flat_type; equal block weights",
            selection="pooled earlier-quarter RMSE; >=1 improvement AND no quarter degradation >1",
            blend_weights=[.25, .5, 1.], primary_used_for_selection=False, primary_previously_inspected=True,
            chronology="exclude entire current training month; query snapshot frozen at training origin",
            unit_loss="rate target; area squared weights normalized to mean 1, also for early stopping",
            test_data_read=False, submission_generated=False,
            limits=["reused historical quarters are not pristine holdouts", "12-month history can be sparse",
                    "static housing coordinates", "6-month diagnostic does not represent full test horizon"])
        self.run_id = signature(self.manifest)
        manifest_path = self.output / "experiment_manifest.json"
        if manifest_path.exists() and json.loads(manifest_path.read_text())["run_id"] != self.run_id:
            raise ValueError("Changed experiment: use a fresh output directory")
        dump(manifest_path, dict(self.manifest, run_id=self.run_id))
        self.cache = ROOT / "models/neighborhood_cache" / self.run_id
        self.cache.mkdir(parents=True, exist_ok=True)
        self.feature_cache = ROOT / "models/neighborhood_features" / signature(dict(
            implementation=sha(ROOT / "src/features/build_neighborhood_features.py"),
            environment=self.manifest["environment"], window=12))
        self.data = load_training(source)
        area_values(self.data)
        self.rows, self.predictions, self.details = [], {}, []

    def frames(self, split):
        start, end = [pd.Period(s) for s in SPLITS[split]]
        months = self.data.RENT_APPROVAL_DATE.dt.to_period("M")
        return self.data.loc[months.lt(start)].copy(), self.data.loc[months.between(start, end)].copy()

    def prepare(self, fit, query, spec):
        if spec["features"]:
            return causal_features(fit, query, cache=self.feature_cache)
        return fit, query.drop(columns=TARGET, errors="ignore")

    def component(self, split, name):
        key = (split, name)
        if key in self.predictions:
            return self.predictions[key]
        fit, query = self.frames(split)
        spec = self.specs[name]
        directory = self.cache / split / name
        meta = directory / "result.json"
        if meta.exists():
            detail = json.loads(meta.read_text())
            with np.load(directory / "prediction.npz") as saved:
                np.testing.assert_array_equal(saved["source_row"], query.index)
                prediction = saved["prediction"].copy()
        else:
            print(f"START {split} {name}", flush=True)
            start = time.perf_counter()
            inner_fit, inner_valid = split_inner(fit)
            xfit, xvalid = self.prepare(inner_fit, inner_valid, spec)
            # Inner labels enter early stopping only, never feature construction.
            xvalid = xvalid.assign(**{TARGET: inner_valid[TARGET]})
            tuned = train(spec, xfit, self.threads, valid=xvalid)
            rounds = chosen_rounds(tuned)
            inner_score = metrics(inner_valid[TARGET], predict(tuned, xvalid, spec["unit"]))
            del tuned, xfit, xvalid
            xfit, xquery = self.prepare(fit, query, spec)
            bundle = train(spec, xfit, self.threads, rounds=rounds)
            prediction = predict(bundle, xquery, spec["unit"])
            detail = dict(split=split, component=name, iterations=rounds, hit_cap=rounds == spec["cap"],
                inner=inner_score, training=metrics(fit[TARGET], predict(bundle, xfit, spec["unit"])),
                outer=metrics(query[TARGET], prediction), fit_rows=len(fit), query_rows=len(query),
                fit_end=str(fit.RENT_APPROVAL_DATE.max().date()),
                inner_fit_end=str(inner_fit.RENT_APPROVAL_DATE.max().date()),
                inner_valid_start=str(inner_valid.RENT_APPROVAL_DATE.min().date()),
                inner_valid_end=str(inner_valid.RENT_APPROVAL_DATE.max().date()),
                seconds=time.perf_counter() - start)
            directory.mkdir(parents=True, exist_ok=True)
            joblib.dump(dict(bundle=bundle, spec=spec, detail=detail), directory / "model.joblib")
            np.savez_compressed(directory / "prediction.npz", source_row=query.index, prediction=prediction)
            dump(meta, detail)
        if name == "housing_control":
            old = OLD_CACHE / f"pred_{split}_lightgbm_cap1000_w0_h0_hist0_direct.npz"
            with np.load(old) as saved:
                np.testing.assert_array_equal(saved["source_row"], query.index)
                np.testing.assert_allclose(prediction, saved["prediction"], atol=1e-8, rtol=1e-10)
        if not np.isfinite(prediction).all() or len(prediction) != len(query):
            raise ValueError("Invalid model prediction")
        self.details.append(detail)
        self.predictions[key] = prediction
        print(f"DONE {split} {name} rmse={detail['outer']['rmse']:.4f} rounds={detail['iterations']}", flush=True)
        return prediction

    def incumbent(self, split):
        _, query = self.frames(split)
        previous = pd.read_csv(OLD / f"predictions_{split}.csv")
        np.testing.assert_array_equal(previous.source_row, query.index)
        np.testing.assert_allclose(previous.actual, query[TARGET], atol=0, rtol=0)
        return previous[CONTROL].to_numpy()

    def recipe(self, name):
        if name == "incumbent":
            return None, 0.
        component, percent = name.rsplit("_blend", 1)
        return component, int(percent) / 100.

    def evaluate(self, split, name):
        key = (split, name)
        if key in self.predictions:
            return
        base = self.incumbent(split)
        component, weight = self.recipe(name)
        prediction = base if component is None else (1-weight)*base + weight*self.component(split, component)
        _, query = self.frames(split)
        self.predictions[key] = prediction
        self.rows.append(dict(split=split, candidate=name, rows=len(query), **metrics(query[TARGET], prediction)))

    def save(self):
        pd.DataFrame(self.rows).to_csv(self.output / "results.csv", index=False)
        dump(self.output / "fit_details.json", self.details)
        for split in {s for s, _ in self.predictions}:
            _, query = self.frames(split)
            table = pd.DataFrame(dict(source_row=query.index, actual=query[TARGET].to_numpy()))
            for (s, name), values in self.predictions.items():
                if s == split:
                    table[name] = values
            table.to_csv(self.output / f"predictions_{split}.csv", index=False)

    def run(self):
        for split in EARLIER:
            self.evaluate(split, "incumbent")
            # The exact old LightGBM component is refitted as an implementation control.
            for name in self.specs:
                self.component(split, name)
                if name != "housing_control":
                    for percent in [25, 50, 100]:
                        self.evaluate(split, f"{name}_blend{percent}")
                self.save()
        table, selected, challenger = choose(self.rows)
        table.to_csv(self.output / "selection.csv", index=False)
        lock = dict(run_id=self.run_id, selected=selected, challenger=challenger,
                    selection_splits=EARLIER, primary_used=False, minimum_gain=1., max_quarter_loss=1.)
        path = self.output / "selection_locked.json"
        if path.exists() and json.loads(path.read_text()) != lock:
            raise ValueError("Selection lock changed")
        dump(path, lock)
        print(f"LOCKED selected={selected} challenger={challenger}", flush=True)
        for split in ["primary_2025", "long_6m"]:
            for name in dict.fromkeys(["incumbent", selected, challenger]):
                self.evaluate(split, name)
            self.save()
        dump(self.output / "completion.json", dict(run_id=self.run_id, completed=True, test_read=False,
                                                  selected=selected, challenger=challenger))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/experiments/neighborhood_v1")
    parser.add_argument("--threads", type=int, default=3)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    Experiment(args.output_dir, args.threads).run()


if __name__ == "__main__":
    main()
