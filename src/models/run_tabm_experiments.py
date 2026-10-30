"""Official TabM-based bounded temporal experiment; no test set or submission."""
from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if __package__ in {None, ""}:
    sys.path.insert(0, str(ROOT))
from src.models.tabm_rent import train, load_bundle, torch_api
from src.models.run_keras_experiments import Experiment as NeuralExperiment, REFERENCE, SEEDS, WEIGHTS
from src.models.run_stage5 import load_training, metrics
from src.models.run_rmse_improvement import EARLIER, SPLITS, dump, sha, signature
from src.models.run_tree_tuning import split_inner
from src.models.train_baseline import TARGET


def specs():
    return {kind: dict(k=32, n_blocks=2, d_block=128, dropout=.1, n_bins=48, d_embedding=8,
        learning_rate=.002, weight_decay=.0003, batch_size=1024, evaluation_batch_size=1024,
        max_epochs=100, patience=12) for kind in ["direct", "trend"]}


class Experiment(NeuralExperiment):
    def __init__(self, output, device):
        self.device = device
        torch = torch_api(device)
        self.output = output.resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        source = ROOT / "data/processed/train_location_features.csv"
        code = [Path(__file__), ROOT / "src/models/tabm_rent.py", ROOT / "requirements-tabm.txt",
            ROOT / "src/models/keras_rent.py", ROOT / "src/models/run_keras_experiments.py",
            ROOT / "src/features/build_temporal_features.py", ROOT / "src/features/build_stage5_features.py",
            ROOT / "src/models/run_stage5.py", ROOT / "src/models/run_rmse_improvement.py",
            ROOT / "src/models/run_tree_tuning.py", ROOT / "src/models/train_baseline.py"]
        environment = {name: importlib.metadata.version(name) for name in
            ["torch", "tabm", "rtdl-num-embeddings", "numpy", "pandas", "scikit-learn"]}
        self.manifest = dict(source=str(source.relative_to(ROOT)), source_sha256=sha(source),
            implementation_sha256={str(p.relative_to(ROOT)): sha(p) for p in code},
            reference_sha256={str(p.relative_to(ROOT)): sha(p) for p in
                [REFERENCE / "selection_locked.json", *[REFERENCE / f"predictions_{s}.csv" for s in SPLITS]]},
            environment=dict(python=platform.python_version(), machine=platform.machine(), **environment),
            backend="pytorch", device=device, cpu_threads=3, grid=specs(), seeds=SEEDS,
            architecture="official TabM BatchEnsemble backbone and LinearEnsemble head; compact learned categorical embeddings as custom inputs",
            numeric_embedding="train-only quantile PiecewiseLinearEmbeddings, version B; constants and missing indicators remain scalar",
            loss="mean squared loss of individual branches, not squared loss of mean prediction",
            seed_ensemble="equal mean of three seeds, each averaging its 32 branches at inference",
            bounded_configuration="two 128-wide blocks and 8-dimensional numeric embeddings, not an exhaustive architecture search",
            training_reuse="long_6m reuses q3_2024 models only after verifying identical training source rows; query predictions are recomputed",
            early_stopping="ensemble scaled MSE on fit-only last 3 months; refit all prior rows at best epoch",
            target="fit-only scaled SGD rent or static residual plus damped24 trend",
            selection_splits=EARLIER, splits=SPLITS, weights_percent=WEIGHTS,
            promotion_gate="at least 1 pooled RMSE improvement; no earlier quarter loses more than 1; no automatic replacement",
            diagnostics=["primary_2025", "long_6m", "long_12m"],
            diagnostics_menu="both three-seed pure models and one locked best mixture, plus incumbent",
            primary_used_for_selection=False, previously_inspected_windows=True,
            bitwise_gpu_reproducibility_guaranteed=False, reload_tolerance_sgd=.02,
            test_data_read=False, submission_generated=False, production_replaced=False,
            readme_modified=False, kaggle_submitted=False,
            official_sources=["https://github.com/yandex-research/tabm", "https://github.com/yandex-research/rtdl-num-embeddings"])
        self.run_id = signature(self.manifest)
        path = self.output / "experiment_manifest.json"
        if path.exists() and json.loads(path.read_text())["run_id"] != self.run_id:
            raise ValueError("Changed experiment; choose a new output directory")
        dump(path, dict(self.manifest, run_id=self.run_id))
        self.cache = ROOT / "models/tabm_cache" / self.run_id
        self.data = load_training(source)
        self.rows, self.details, self.seed_rows, self.predictions, self.components = [], [], [], {}, {}

    def component(self, split, kind):
        key = split, kind
        if key in self.components:
            return self.components[key]
        fit, query = self.frames(split)
        predictions = []
        for seed in SEEDS:
            directory = self.cache / split / kind / f"seed_{seed}"
            shared = self.cache / "q3_2024" / kind / f"seed_{seed}"
            if split == "long_6m" and not (directory / "result.json").exists():
                q3_fit, _ = self.frames("q3_2024")
                np.testing.assert_array_equal(q3_fit.index, fit.index)
                old = json.loads((shared / "result.json").read_text())
                for file, expected in old["saved_sha256"].items():
                    if sha(shared / file) != expected:
                        raise ValueError("Changed shared-origin TabM model")
                bundle = load_bundle(shared)
                pred = bundle.predict(query)
                directory.mkdir(parents=True, exist_ok=True)
                for file in ["model.pt", "schema.json"]:
                    shutil.copy2(shared / file, directory / file)
                np.savez_compressed(directory / "prediction.npz", source_row=query.index, prediction=pred)
                detail = dict(old, split=split, query_rows=len(query), outer=metrics(query[TARGET], pred),
                    reused_training_from=str(shared.relative_to(ROOT)), seconds=0.,
                    saved_sha256={p: sha(directory / p) for p in ["model.pt", "schema.json", "prediction.npz"]})
                dump(directory / "result.json", detail)
                del bundle
                gc.collect()
            if (directory / "result.json").exists():
                detail = json.loads((directory / "result.json").read_text())
                for file, expected in detail["saved_sha256"].items():
                    if sha(directory / file) != expected:
                        raise ValueError("Changed TabM cache")
                with np.load(directory / "prediction.npz") as stored:
                    np.testing.assert_array_equal(stored["source_row"], query.index)
                    pred = stored["prediction"].copy()
            else:
                print(f"START {split} {kind} seed={seed} device={self.device}", flush=True)
                started = time.perf_counter()
                inner_fit, inner_valid = split_inner(fit)
                tuned, inner_history = train(inner_fit, kind, seed, self.manifest["grid"][kind],
                    valid=inner_valid, label=f"{split}/{kind}/{seed}/inner", device=self.device)
                chosen = inner_history["best_epoch"]
                inner_metrics = metrics(inner_valid[TARGET], tuned.predict(inner_valid))
                del tuned
                gc.collect()
                bundle, refit_history = train(fit, kind, seed, self.manifest["grid"][kind], epochs=chosen,
                    label=f"{split}/{kind}/{seed}/refit", device=self.device)
                pred = bundle.predict(query)
                training_metrics = metrics(fit[TARGET], bundle.predict(fit))
                bundle.save(directory)
                loaded = load_bundle(directory)
                reloaded = loaded.predict(query)
                difference = float(np.max(np.abs(pred - reloaded)))
                np.testing.assert_allclose(pred, reloaded, atol=.02, rtol=0)
                np.testing.assert_allclose(pred, loaded.predict(query.assign(MONTHLY_RENT=-999999., Id=987654)), atol=.02, rtol=0)
                np.savez_compressed(directory / "prediction.npz", source_row=query.index, prediction=pred)
                detail = dict(split=split, component=kind, seed=seed, best_epoch=chosen,
                    fit_rows=len(fit), query_rows=len(query), fit_end=str(fit.RENT_APPROVAL_DATE.max().date()),
                    inner_fit_end=str(inner_fit.RENT_APPROVAL_DATE.max().date()),
                    inner_valid_start=str(inner_valid.RENT_APPROVAL_DATE.min().date()),
                    inner_valid_end=str(inner_valid.RENT_APPROVAL_DATE.max().date()),
                    inner=inner_metrics, training=training_metrics, outer=metrics(query[TARGET], pred),
                    inner_history=inner_history, refit_history=refit_history,
                    model_reload_verified=True, maximum_reload_difference_sgd=difference,
                    query_target_and_id_invariance=True, seconds=time.perf_counter()-started,
                    saved_sha256={p: sha(directory / p) for p in ["model.pt", "schema.json", "prediction.npz"]})
                dump(directory / "result.json", detail)
                del bundle, loaded
                gc.collect()
                if self.device == "mps":
                    torch_api(self.device).mps.empty_cache()
            self.details.append(detail)
            self.seed_rows.append(dict(split=split, component=kind, seed=seed, rows=len(query),
                best_epoch=detail["best_epoch"], **metrics(query[TARGET], pred)))
            predictions.append(pred)
            self.save()
            print(f"DONE {split} {kind} seed={seed} RMSE={detail['outer']['rmse']:.4f} epochs={detail['best_epoch']}", flush=True)
        self.components[key] = np.mean(predictions, axis=0)
        return self.components[key]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/experiments/tabm_v1")
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--selection-only", action="store_true")
    args = parser.parse_args()
    Experiment(args.output_dir, args.device).run(args.selection_only)
