"""Bounded embedding MLP and trend-residual experiments, three seeds each.

Nested temporal early stopping; no test data, submission or automatic promotion.
"""
from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if __package__ in {None, ""}:
    sys.path.insert(0, str(ROOT))
from src.models.keras_rent import train, load_bundle, recipe_schema, release
from src.models.run_stage5 import load_training, metrics
from src.models.run_rmse_improvement import EARLIER, SPLITS, dump, sha, signature
from src.models.run_tree_tuning import split_inner
from src.models.train_baseline import TARGET

REFERENCE = ROOT / "reports/experiments/rmse_improvement_v1"
CONTROL = "trend_damped24_weight25"
SEEDS = [42, 123, 2026]
WEIGHTS = [10, 25, 50, 100]


def specs():
    return {name: dict(widths=[128, 64], dropout=.1, dense_l2=1e-5, embedding_l2=1e-6,
        learning_rate=.002, batch_size=1024, max_epochs=80, patience=10) for name in ["direct", "trend"]}


def ranking(rows):
    table = pd.DataFrame(rows)
    table = table.loc[table.split.isin(EARLIER)].copy()
    if table.duplicated(["split", "candidate"]).any() or table.empty:
        raise ValueError("Selection rows must be complete and unique")
    if not table.groupby("candidate").split.agg(set).map(lambda x: x == set(EARLIER)).all():
        raise ValueError("Every candidate needs both earlier quarters")
    reference = table.loc[table.candidate.eq("incumbent")].set_index("split").rmse
    if set(reference.index) != set(EARLIER):
        raise ValueError("Missing incumbent")
    result = []
    base = float(np.sqrt(np.average(reference.to_numpy() ** 2,
        weights=table.loc[table.candidate.eq("incumbent")].set_index("split").loc[reference.index, "rows"])))
    for candidate, group in table.groupby("candidate"):
        pooled = float(np.sqrt(np.average(group.rmse ** 2, weights=group.rows)))
        worst = float(max(r.rmse - reference[r.split] for r in group.itertuples()))
        result.append(dict(candidate=candidate, rows=int(group.rows.sum()), pooled_rmse=pooled,
            gain=base - pooled, worst_quarter_delta=worst, eligible=base - pooled >= 1. and worst <= 1.))
    scores = pd.DataFrame(result).sort_values(["pooled_rmse", "candidate"]).reset_index(drop=True)
    pure = str(scores.loc[scores.candidate.str.endswith("_blend100")].iloc[0].candidate)
    new = str(scores.loc[~scores.candidate.eq("incumbent")].iloc[0].candidate)
    eligible = scores.loc[scores.eligible]
    selected = str(eligible.iloc[0].candidate) if len(eligible) else "incumbent"
    return scores, pure, new, selected


class Experiment:
    def __init__(self, output):
        self.output = output.resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        source = ROOT / "data/processed/train_location_features.csv"
        code = [Path(__file__), ROOT / "src/models/keras_rent.py", ROOT / "requirements-keras.txt",
            ROOT / "src/features/build_temporal_features.py", ROOT / "src/features/build_stage5_features.py",
            ROOT / "src/models/run_stage5.py", ROOT / "src/models/run_rmse_improvement.py",
            ROOT / "src/models/run_tree_tuning.py", ROOT / "src/models/train_baseline.py"]
        environment = {name: importlib.metadata.version(name) for name in
            ["keras", "jax", "jaxlib", "numpy", "pandas", "scikit-learn", "h5py", "ml-dtypes"]}
        self.manifest = dict(source=str(source.relative_to(ROOT)), source_sha256=sha(source),
            implementation_sha256={str(p.relative_to(ROOT)): sha(p) for p in code},
            reference_sha256={str(p.relative_to(ROOT)): sha(p) for p in
                [REFERENCE / "selection_locked.json", *[REFERENCE / f"predictions_{s}.csv" for s in SPLITS]]},
            environment=dict(python=platform.python_version(), machine=platform.machine(), **environment),
            backend="jax", device="cpu", grid=specs(), seeds=SEEDS, seed_ensemble="equal mean of 3 predictions",
            selection_splits=EARLIER, splits=SPLITS, weights_percent=WEIGHTS,
            preprocessing="fit-only rare-category vocabulary, median imputation, standardization and missing indicators",
            target="fit-only centered/scaled original SGD rent or static residual plus damped24 trend; MSE",
            early_stopping="fit-only last 3 months; scaled MSE excluding regularization; refit all prior rows at best epoch",
            promotion_gate="at least 1 pooled RMSE improvement, no earlier-quarter loss exceeding 1; no automatic replacement",
            diagnostics=["primary_2025", "long_6m", "long_12m"],
            diagnostics_menu="both three-seed pure networks and one locked best mixture, plus incumbent",
            primary_used_for_selection=False, previously_inspected_windows=True,
            test_data_read=False, submission_generated=False, production_replaced=False,
            official_sources=["https://keras.io/getting_started/", "https://keras.io/api/callbacks/early_stopping/"])
        self.run_id = signature(self.manifest)
        path = self.output / "experiment_manifest.json"
        if path.exists() and json.loads(path.read_text())["run_id"] != self.run_id:
            raise ValueError("Output belongs to another experiment; choose a new directory")
        dump(path, dict(self.manifest, run_id=self.run_id))
        self.cache = ROOT / "models/keras_cache" / self.run_id
        self.data = load_training(source)
        self.rows, self.details, self.seed_rows, self.predictions, self.components = [], [], [], {}, {}

    def frames(self, split):
        start, end = [pd.Period(s) for s in SPLITS[split]]
        months = self.data.RENT_APPROVAL_DATE.dt.to_period("M")
        return self.data.loc[months.lt(start)].copy(), self.data.loc[months.between(start, end)].copy()

    def component(self, split, kind):
        key = (split, kind)
        if key in self.components:
            return self.components[key]
        fit, query = self.frames(split)
        predictions = []
        for seed in SEEDS:
            directory = self.cache / split / kind / f"seed_{seed}"
            if (directory / "result.json").exists():
                detail = json.loads((directory / "result.json").read_text())
                for file, expected in detail["saved_sha256"].items():
                    if sha(directory / file) != expected:
                        raise ValueError("Changed model cache")
                with np.load(directory / "prediction.npz") as stored:
                    np.testing.assert_array_equal(stored["source_row"], query.index)
                    pred = stored["prediction"].copy()
            else:
                print(f"START {split} {kind} seed={seed}", flush=True)
                started = time.perf_counter()
                inner_fit, inner_valid = split_inner(fit)
                tuned, inner_history = train(inner_fit, kind, seed, self.manifest["grid"][kind],
                    valid=inner_valid, label=f"{split}/{kind}/{seed}/inner")
                chosen = inner_history["best_epoch"]
                inner_metrics = metrics(inner_valid[TARGET], tuned.predict(inner_valid))
                del tuned
                gc.collect()
                bundle, refit_history = train(fit, kind, seed, self.manifest["grid"][kind], epochs=chosen,
                    label=f"{split}/{kind}/{seed}/refit")
                pred = bundle.predict(query)
                training_metrics = metrics(fit[TARGET], bundle.predict(fit))
                bundle.save(directory)
                reloaded = load_bundle(directory)
                reload_prediction = reloaded.predict(query)
                np.testing.assert_allclose(pred, reload_prediction, atol=1e-5, rtol=0)
                np.testing.assert_allclose(pred, reloaded.predict(query.assign(MONTHLY_RENT=-999999., Id=987654)),
                                           atol=1e-5, rtol=0)
                np.savez_compressed(directory / "prediction.npz", source_row=query.index, prediction=pred)
                detail = dict(split=split, component=kind, seed=seed, best_epoch=chosen,
                    fit_rows=len(fit), query_rows=len(query), fit_end=str(fit.RENT_APPROVAL_DATE.max().date()),
                    inner_fit_end=str(inner_fit.RENT_APPROVAL_DATE.max().date()),
                    inner_valid_start=str(inner_valid.RENT_APPROVAL_DATE.min().date()),
                    inner_valid_end=str(inner_valid.RENT_APPROVAL_DATE.max().date()),
                    inner=inner_metrics, training=training_metrics, outer=metrics(query[TARGET], pred),
                    inner_history=inner_history, refit_history=refit_history,
                    model_reload_verified=True, query_target_and_id_invariance=True,
                    seconds=time.perf_counter() - started,
                    saved_sha256={p: sha(directory / p) for p in ["model.keras", "schema.json", "prediction.npz"]})
                dump(directory / "result.json", detail)
                del reloaded
                release(bundle)
                del bundle
            self.details.append(detail)
            self.seed_rows.append(dict(split=split, component=kind, seed=seed, rows=len(query),
                best_epoch=detail["best_epoch"], **metrics(query[TARGET], pred)))
            predictions.append(pred)
            self.save()
            print(f"DONE {split} {kind} seed={seed} RMSE={detail['outer']['rmse']:.4f} epochs={detail['best_epoch']}", flush=True)
        self.components[key] = np.mean(predictions, axis=0)
        return self.components[key]

    def evaluate(self, split, candidate):
        key = split, candidate
        if key in self.predictions:
            return
        _, query = self.frames(split)
        old = pd.read_csv(REFERENCE / f"predictions_{split}.csv")
        np.testing.assert_array_equal(old.source_row, query.index)
        np.testing.assert_array_equal(old.actual, query[TARGET])
        reference = old[CONTROL].to_numpy()
        pred = reference.copy()
        if candidate != "incumbent":
            kind, percentage = candidate.rsplit("_blend", 1)
            weight = int(percentage) / 100
            pred = (1 - weight) * reference + weight * self.component(split, kind)
        self.predictions[key] = pred
        self.rows.append(dict(split=split, candidate=candidate, rows=len(query),
            residual_correlation_with_incumbent=float(np.corrcoef(pred-query[TARGET], reference-query[TARGET])[0, 1]),
            **metrics(query[TARGET], pred)))

    def save(self):
        if self.rows:
            pd.DataFrame(self.rows).to_csv(self.output / "results.csv", index=False)
        if self.seed_rows:
            pd.DataFrame(self.seed_rows).to_csv(self.output / "seed_results.csv", index=False)
        dump(self.output / "fit_details.json", self.details)
        for split in sorted({s for s, _ in self.predictions}):
            _, query = self.frames(split)
            table = pd.DataFrame(dict(source_row=query.index, actual=query[TARGET].to_numpy()))
            for (s, name), pred in self.predictions.items():
                if s == split:
                    table[name] = pred
            table.to_csv(self.output / f"predictions_{split}.csv", index=False)

    def run(self, selection_only=False):
        dump(self.output / "completion.json", dict(completed=False, run_id=self.run_id,
            test_data_read=False, submission_generated=False, production_replaced=False))
        for split in EARLIER:
            self.evaluate(split, "incumbent")
            for kind in self.manifest["grid"]:
                for percentage in WEIGHTS:
                    self.evaluate(split, f"{kind}_blend{percentage}")
                self.save()
        scores, pure, new, selected = ranking(self.rows)
        scores.to_csv(self.output / "selection.csv", index=False)
        lock = dict(run_id=self.run_id, best_pure=pure, best_new=new, selected=selected,
            selection_splits=EARLIER, primary_used=False, production_replaced=False)
        path = self.output / "selection_locked.json"
        if path.exists() and json.loads(path.read_text()) != lock:
            raise ValueError("Selection lock changed")
        dump(path, lock)
        print(f"LOCKED {json.dumps(lock)}", flush=True)
        if selection_only:
            return
        for split in self.manifest["diagnostics"]:
            for candidate in dict.fromkeys(["incumbent", "direct_blend100", "trend_blend100", new]):
                self.evaluate(split, candidate)
                self.save()
        dump(self.output / "completion.json", dict(completed=True, **lock,
            score_rows=len(self.rows), fitted_models=len(self.details),
            test_data_read=False, submission_generated=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/experiments/keras_v1")
    parser.add_argument("--selection-only", action="store_true")
    args = parser.parse_args()
    Experiment(args.output_dir).run(args.selection_only)
