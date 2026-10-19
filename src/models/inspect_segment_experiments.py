"""Independently check segment experiment scores, locks and saved models."""
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

from src.models.run_stage5 import load_training
from src.models.run_rmse_improvement import SPLITS, sha, dump
from src.models.run_segment_experiments import BASE, CALIBRATION, CALIB_SOURCES
from src.features.build_stage5_features import aggregate_macro, attach_macro
from src.features.segment_calibration import available_macro


def audit(output: Path) -> dict:
    output = output.resolve()
    manifest = json.loads((output / "experiment_manifest.json").read_text())
    lock = json.loads((output / "selection_locked.json").read_text())
    assert manifest["complete"] and not manifest["test_data_read"] and not manifest["submission_generated"]
    assert not lock["primary_used"] and not manifest["primary_used_for_selection"]
    for file, expected in manifest["implementation_sha256"].items():
        assert sha(ROOT / file) == expected, file
    source = ROOT / "data/processed/train_location_features.csv"
    assert sha(source) == manifest["source_sha256"]
    for file, expected in manifest["macro_sha256"].items():
        assert sha(ROOT / file) == expected, file
    train = load_training(source)
    month = train.RENT_APPROVAL_DATE.dt.to_period("M")
    splits = {}
    for name, (start, end) in {**SPLITS, **CALIBRATION}.items():
        splits[name] = (train.loc[month.lt(pd.Period(start))], train.loc[month.between(pd.Period(start), pd.Period(end))])
    base_meta = json.loads((output / "base_predictions/experiment_manifest.json").read_text())
    base_cache = ROOT / "models/rmse_improvement_cache" / base_meta["run_id"]
    old_lock = json.loads((ROOT / "reports/experiments/rmse_improvement_v1/selection_locked.json").read_text())
    monthly, _ = aggregate_macro(pd.read_csv(ROOT / "data/interim/auxiliary/sg-coe-prices_clean.csv"),
        pd.read_csv(ROOT / "data/interim/auxiliary/sg-stock-prices_clean.csv"))
    cache = Path(manifest["cache"])
    predictions = {}
    native_models_reloaded = set()

    def predict(split, name):
        key = (split, name)
        if key in predictions:
            return predictions[key]
        fit, query = splits[split]
        recipe = lock["grid"][name]
        if recipe["kind"] == "base":
            value = np.zeros(len(query))
            for component, weight in old_lock["weights"].items():
                with np.load(base_cache / f"pred_{split}_{component}.npz") as saved:
                    np.testing.assert_array_equal(saved["source_row"], query.index)
                    value += weight * saved["prediction"]
        elif recipe["kind"] == "correction":
            model = joblib.load(cache / f"correction_{split}_{name}.joblib")
            months = (query.RENT_APPROVAL_DATE.dt.year * 12 + query.RENT_APPROVAL_DATE.dt.month).to_numpy()
            assert months.min() > model["cutoff"]
            correction = []
            for index, row in enumerate(query.to_dict("records")):
                group = tuple(str(row[c]) for c in model["columns"])
                level, slope, _ = model["params"].get(group, (0, 0, 0))
                h = months[index] - model["cutoff"]
                correction.append(np.clip(level + slope * 6 * (1 - np.exp(-h / 6)), -200, 200))
            value = predict(split, BASE) + np.asarray(correction)
        else:
            source = recipe["source"]
            if source == "expert":
                directory = cache / f"expert_{split}"
                other = predict(split, BASE).copy()
                for details in json.loads((directory / "fit.json").read_text()):
                    group = details["group"]
                    assert details["rows"] == int(fit.FLAT_TYPE.eq(group).sum())
                    assert details["rows"] >= 3000 and details["inner_rows"] >= 100
                    path = directory / f"{group}.joblib"
                    model = joblib.load(path)
                    mask = query.FLAT_TYPE.eq(group)
                    other[mask] = model.predict(query.loc[mask].drop(columns="MONTHLY_RENT"))
                    native_models_reloaded.add(str(path))
            else:
                directory = cache / f"macro_{split}_{source}"
                path = directory / "model.joblib"
                model = joblib.load(path)
                data = query.drop(columns="MONTHLY_RENT")
                data = attach_macro(data, monthly, cutoff=fit.RENT_APPROVAL_DATE.max()) if source == "market_frozen2" else available_macro(data, monthly)
                other = model.predict(data)
                native_models_reloaded.add(str(path))
            with np.load(directory / "prediction.npz") as saved:
                np.testing.assert_array_equal(saved["source_row"], query.index)
                np.testing.assert_allclose(other, saved["prediction"], rtol=0, atol=1e-8)
            value = predict(split, recipe["incumbent"]) * (1 - recipe["weight"]) + other * recipe["weight"]
        predictions[key] = np.maximum(value, 0)
        return predictions[key]

    results = pd.read_csv(output / "results.csv")
    assert not results.duplicated(["split", "candidate"]).any()
    for row in results.itertuples():
        query = splits[row.split][1]
        value = predict(row.split, row.candidate)
        error = value - query.MONTHLY_RENT.to_numpy()
        assert row.validation_rows == len(query)
        np.testing.assert_allclose([row.rmse, row.mae, row.bias],
            [np.sqrt(np.mean(error ** 2)), np.mean(np.abs(error)), np.mean(error)], rtol=0, atol=1e-7)
    for decision in lock["decisions"]:
        table = results.loc[results.split.isin(["q3_2024", "q4_2024"]) & results.candidate.isin(decision["candidates"])].copy()
        assert table.groupby("candidate").size().eq(2).all()
        table["sse"] = table.rmse ** 2 * table.validation_rows
        ranked = table.groupby("candidate").agg(sse=("sse", "sum"), n=("validation_rows", "sum"))
        score = np.sqrt(ranked.sse / ranked.n)
        assert score[decision["selected"]] <= score.min() + 1e-8
        np.testing.assert_allclose(score[decision["selected"]], decision["pooled_rmse"], rtol=0, atol=1e-8)
    tails, partitions_checked = {}, 0
    for split in ["q3_2024", "q4_2024", "primary_2025", "long_6m"]:
        query = splits[split][1]
        exported = pd.read_csv(output / f"predictions_{split}.csv")
        np.testing.assert_array_equal(exported.source_row, query.index)
        np.testing.assert_array_equal(exported.actual, query.MONTHLY_RENT)
        for name in exported.columns.difference(["source_row", "actual"]):
            value = predict(split, name)
            np.testing.assert_allclose(value, exported[name], rtol=0, atol=1e-8)
            tag = "control" if name == BASE else "selected"
            groups = pd.read_csv(output / f"segments_{tag}_{split}.csv")
            error = value - query.MONTHLY_RENT.to_numpy()
            for _, partition in groups.groupby("partition"):
                assert partition.rows.sum() == len(query)
                np.testing.assert_allclose(partition.sse.sum(), np.dot(error, error), rtol=1e-12)
                np.testing.assert_allclose(partition.sse_share.sum(), 1., atol=1e-10)
                assert (partition.bias_sse_share <= partition.sse_share + 1e-12).all()
                partitions_checked += 1
        original = pd.read_csv(ROOT / f"reports/experiments/rmse_improvement_v1/predictions_{split}.csv")
        np.testing.assert_allclose(exported[BASE], original[old_lock["selected"]], rtol=0, atol=1e-8)
        errors = (predict(split, BASE) - query.MONTHLY_RENT.to_numpy()) ** 2
        descending = np.sort(errors)[::-1]
        tails[split] = {f"top_{int(fraction * 100)}pct_sse_share": float(descending[:max(1, int(np.ceil(len(errors) * fraction)))].sum() / errors.sum()) for fraction in [.01, .05, .10]}
        provenance = json.loads((output / f"calibration_{split}.json").read_text())
        for entry in provenance:
            assert entry["fit_end"] < entry["query_start"] <= entry["query_end"] < f"{query.RENT_APPROVAL_DATE.min():%Y-%m}"
    audit_result = dict(passed=True, scores_checked=len(results), partitions_checked=partitions_checked,
        native_models_reloaded=len(native_models_reloaded), control_reproduced=True,
        selection_excludes_primary=True, selected=lock["selected"], tail_error_concentration=tails)
    dump(output / "output_validation.json", audit_result)
    print(json.dumps(audit_result, indent=2))
    return audit_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/experiments/segment_v1")
    audit(parser.parse_args().output_dir)
