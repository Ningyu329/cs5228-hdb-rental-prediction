"""Independently recompute exported scores, locks, row order and model predictions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if __package__ in {None, ""}:
    sys.path.insert(0, str(ROOT))

from src.features.build_temporal_features import trend_values
from src.models.run_rmse_improvement import Spec, prepare, select_training, sha, signature, dump
from src.models.run_stage5 import load_training


def audit(output: Path) -> dict:
    output = output.resolve()
    manifest = json.loads((output / "experiment_manifest.json").read_text())
    if not manifest.get("complete"):
        raise ValueError("Experiment has not finished")
    source = ROOT / manifest["source"]
    assert sha(source) == manifest["source_sha256"], "Input changed"
    for file, expected in manifest["implementation_sha256"].items():
        assert sha(ROOT / file) == expected, file
    train = load_training(source)
    months = train.RENT_APPROVAL_DATE.dt.to_period("M")
    results = pd.read_csv(output / "results.csv")
    assert not results.duplicated(["split", "candidate"]).any()
    lock = json.loads((output / "selection_locked.json").read_text())
    assert not lock["primary_used"] and not manifest["primary_used_for_selection"]
    score_count, prediction_count, reload_count = 0, 0, 0
    for split, (start, end) in manifest["splits"].items():
        first, last = pd.Period(start), pd.Period(end)
        fit = train.loc[months.lt(first)].copy()
        query = train.loc[months.between(first, last)].copy()
        frame = pd.read_csv(output / f"predictions_{split}.csv")
        np.testing.assert_array_equal(frame.source_row, query.index)
        np.testing.assert_allclose(frame.actual, query.MONTHLY_RENT, rtol=0, atol=1e-8)
        for name in frame.columns.difference(["source_row", "actual"]):
            prediction = frame[name].to_numpy()
            assert np.isfinite(prediction).all() and (prediction >= 0).all()
            error = prediction - frame.actual.to_numpy()
            expected = dict(rmse=np.sqrt(np.mean(error ** 2)), mae=np.mean(np.abs(error)), bias=np.mean(error))
            row = results.loc[results.split.eq(split) & results.candidate.eq(name)]
            assert len(row) == 1, (split, name)
            for metric, value in expected.items():
                np.testing.assert_allclose(row.iloc[0][metric], value, rtol=0, atol=1e-7)
            score_count += 1
            prediction_count += len(prediction)
        if split == "primary_2025":
            reloaded = {}
            cache = ROOT / "models/rmse_improvement_cache" / manifest["run_id"]
            # Also check control components independently of selected choice.
            control = [Spec("catboost", 800), Spec("lightgbm", 1000)]
            specs = {}
            for path in output.glob(f"fit_{split}_*.json"):
                spec = Spec(**json.loads(path.read_text())["spec"])
                specs[spec.name] = spec
            for name, spec in specs.items():
                selected = select_training(fit, spec)
                meta = json.loads((output / f"fit_{split}_{name}.json").read_text())
                recipe = {k: v for k, v in lock["specs"].get(name, spec.__dict__).items() if k != "cap"}
                key = signature(dict(recipe=recipe, fit_start=str(selected.RENT_APPROVAL_DATE.min()),
                    fit_end=str(selected.RENT_APPROVAL_DATE.max()), rows=len(selected),
                    rounds=meta["iterations"], early=False))
                payload = joblib.load(cache / f"fit_{key}.joblib")
                _, featured, _, _, _ = prepare(selected, query.drop(columns="MONTHLY_RENT"), spec)
                pred = np.asarray(payload["bundle"].model.predict(payload["bundle"].matrix(featured)))
                if payload["trend"] is not None:
                    pred += trend_values(payload["trend"], query)
                pred = np.maximum(pred, 0)
                with np.load(cache / f"pred_{split}_{name}.npz") as stored:
                    np.testing.assert_allclose(pred, stored["prediction"], rtol=0, atol=1e-8)
                reloaded[name] = pred
                reload_count += 1
            selected_prediction = sum(reloaded[name] * weight for name, weight in lock["weights"].items())
            np.testing.assert_allclose(selected_prediction, frame[lock["selected"]], rtol=0, atol=1e-8)
            np.testing.assert_allclose((reloaded[control[0].name] + reloaded[control[1].name]) / 2,
                frame[lock["baseline"]], rtol=0, atol=1e-8)
            # Intermediate checkpoints are scored in results.csv even when only
            # control/final row-level predictions are exported. Reconstruct them
            # from the independently reloaded native components as well.
            pending = results.loc[results.split.eq(split) & ~results.candidate.isin(frame.columns)]
            for row in pending.itertuples():
                if row.candidate.startswith("cap_") and row.candidate.endswith("_blend"):
                    cap = int(row.candidate.split("_")[1])
                    names = [Spec("catboost", cap).name, Spec("lightgbm", 1000).name]
                elif row.candidate == "recency_confirmed_blend":
                    names = [name for name, spec in specs.items() if (spec.window or spec.half_life) and not spec.history and not spec.trend]
                elif row.candidate == "history_confirmed_blend":
                    names = [name for name, spec in specs.items() if spec.history and not spec.trend]
                else:
                    raise ValueError(f"Unknown intermediate checkpoint: {row.candidate}")
                assert len(names) == 2 and {specs[name].family for name in names} == {"catboost", "lightgbm"}
                prediction = (reloaded[names[0]] + reloaded[names[1]]) / 2
                error = prediction - query.MONTHLY_RENT.to_numpy()
                for metric, value in dict(rmse=np.sqrt(np.mean(error**2)), mae=np.mean(np.abs(error)), bias=np.mean(error)).items():
                    np.testing.assert_allclose(getattr(row, metric), value, rtol=0, atol=1e-7)
                score_count += 1
                prediction_count += len(prediction)
    for decision in lock["decisions"]:
        rows = results.loc[results.split.isin(["q3_2024", "q4_2024"]) & results.candidate.isin(decision["candidates"])].copy()
        assert rows.groupby("candidate").split.nunique().eq(2).all()
        rows["sse"] = rows.rmse ** 2 * rows.validation_rows
        ranking = rows.groupby("candidate").agg(sse=("sse", "sum"), n=("validation_rows", "sum"))
        ranking["pooled"] = np.sqrt(ranking.sse / ranking.n)
        ranking = ranking.reset_index().sort_values(["pooled", "candidate"])
        assert ranking.iloc[0].candidate == decision["selected"], decision
        np.testing.assert_allclose(ranking.iloc[0].pooled, decision["pooled_rmse"], atol=1e-8, rtol=0)
    historical_path = ROOT / "reports/experiments/tree_tuning_v1/comparison.csv"
    historical = pd.read_csv(historical_path)
    historical_control = float(historical.loc[historical.model.eq("Locked selection"), "rmse"].iloc[0])
    current_control = float(results.loc[results.split.eq("primary_2025") & results.candidate.eq(lock["baseline"]), "rmse"].iloc[0])
    assert score_count == len(results), "Some score rows were not independently checked"
    summary = dict(passed=True, score_rows_checked=score_count, all_result_rows_checked=True, prediction_values_checked=prediction_count,
        primary_native_models_reloaded=reload_count, input_and_code_hashes_checked=True,
        validation_implementation_sha256=sha(Path(__file__)),
        historical_reference_sha256=sha(historical_path),
        control_primary_rmse=current_control, historical_primary_rmse=historical_control,
        control_reproduction_difference=current_control-historical_control,
        stage_locks_checked=len(lock["decisions"]), labels_and_source_rows_aligned=True,
        primary_used_for_selection=False, test_data_read=False,
        note="Long backtests overlap earlier tuning periods and are diagnostics, not untouched holdouts.")
    dump(output / "output_validation.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/experiments/rmse_improvement_v1")
    print(json.dumps(audit(parser.parse_args().output_dir), ensure_ascii=False, indent=2))
