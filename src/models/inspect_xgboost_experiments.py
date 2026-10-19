"""Independent arithmetic, model reload and time-boundary checks for XGBoost."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if __package__ in {None, ""}:
    sys.path.insert(0, str(ROOT))

from src.features.build_neighborhood_features import ALL_FEATURES, snapshot
from src.models.run_stage5 import load_training
from src.models.run_rmse_improvement import EARLIER, SPLITS, sha, dump
from src.models.run_xgboost_experiments import load_bundle, CONTROL


def inspect(output):
    manifest = json.loads((output/"experiment_manifest.json").read_text())
    lock = json.loads((output/"selection_locked.json").read_text())
    complete = json.loads((output/"completion.json").read_text())
    identity = {k: v for k, v in manifest.items() if k != "run_id"}
    assert hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20] == manifest["run_id"]
    assert manifest["run_id"] == lock["run_id"] == complete["run_id"] and complete["completed"]
    assert not manifest["test_data_read"] and not manifest["submission_generated"]
    assert not lock["primary_used"] and lock["selection_splits"] == EARLIER
    assert not lock["production_replaced"] and not manifest["production_replaced"]
    assert sha(ROOT/manifest["source"]) == manifest["source_sha256"]
    for section in ["implementation_sha256", "reference_sha256"]:
        for path, expected in manifest[section].items():
            assert sha(ROOT/path) == expected, path
    data = load_training(ROOT/manifest["source"])
    months = data.RENT_APPROVAL_DATE.dt.to_period("M")
    results = pd.read_csv(output/"results.csv")
    assert not results.duplicated(["split", "candidate"]).any()
    frames, tables = {}, {}
    score_checks = 0
    diagnostic_rows = []
    for split in results.split.unique():
        start, end = [pd.Period(s) for s in SPLITS[split]]
        fit, query = data.loc[months.lt(start)], data.loc[months.between(start, end)]
        frames[split] = fit, query
        table = pd.read_csv(output/f"predictions_{split}.csv")
        tables[split] = table
        np.testing.assert_array_equal(table.source_row, query.index)
        np.testing.assert_array_equal(table.actual, query.MONTHLY_RENT)
        old = pd.read_csv(ROOT/f"reports/experiments/rmse_improvement_v1/predictions_{split}.csv")
        np.testing.assert_allclose(table.incumbent, old[CONTROL], atol=1e-8, rtol=0)
        for row in results.loc[results.split.eq(split)].itertuples():
            pred = table[row.candidate].to_numpy()
            error = pred-table.actual.to_numpy()
            np.testing.assert_allclose([np.sqrt(np.mean(error**2)), np.mean(np.abs(error)), error.mean()],
                [row.rmse, row.mae, row.bias], atol=1e-8, rtol=0)
            assert row.rows == len(query)
            if row.candidate != "incumbent":
                component, percent = row.candidate.rsplit("_blend", 1)
                weight = int(percent)/100
                np.testing.assert_allclose(pred, (1-weight)*table.incumbent+weight*table[component], atol=1e-8, rtol=0)
            score_checks += 1
        for name in dict.fromkeys(["incumbent", lock["best_pure"], lock["best_new"]]):
            error = (table[name]-table.actual).to_numpy()
            for month in sorted(query.RENT_APPROVAL_DATE.dt.strftime("%Y-%m").unique()):
                mask = query.RENT_APPROVAL_DATE.dt.strftime("%Y-%m").eq(month).to_numpy()
                e = error[mask]
                diagnostic_rows.append(dict(split=split, month=month, candidate=name, rows=int(mask.sum()),
                    rmse=float(np.sqrt(np.mean(e**2))), mae=float(np.mean(np.abs(e))), bias=float(e.mean())))
    selection = pd.read_csv(output/"selection.csv").set_index("candidate")
    expected_names = {"incumbent"} | {f"{name}_blend{p}" for name in manifest["grid"] for p in [25, 50, 100]}
    assert set(selection.index) == expected_names
    independently_ranked = []
    for name in expected_names:
        errors = np.concatenate([(tables[s][name]-tables[s].actual).to_numpy() for s in EARLIER])
        score = np.sqrt(np.mean(errors**2))
        row = selection.loc[name]
        np.testing.assert_allclose([row.sse, row.pooled_rmse], [np.sum(errors**2), score], atol=1e-5, rtol=1e-12)
        assert int(row.rows) == len(errors)
        assert bool(row.below_300) == (score < 300) and bool(row.below_200) == (score < 200)
        independently_ranked.append((score, name))
    independently_ranked.sort()
    assert lock["best_pure"] == next(n for _, n in independently_ranked if n.endswith("_blend100"))
    assert lock["best_new"] == next(n for _, n in independently_ranked if n != "incumbent")
    native_checks = 0
    details = json.loads((output/"fit_details.json").read_text())
    assert len(details) == len({(d["split"], d["component"]) for d in details})
    feature_snapshots = {}
    for detail in details:
        split, name = detail["split"], detail["component"]
        fit, query = frames[split]
        directory = ROOT/"models/xgboost_cache"/manifest["run_id"]/split/name
        bundle = load_bundle(directory)
        assert bundle.model.get_booster().num_boosted_rounds() == detail["iterations"]
        assert 1 <= detail["iterations"] <= manifest["grid"][name]["cap"]
        assert pd.Timestamp(detail["inner_fit_end"]) < pd.Timestamp(detail["inner_valid_start"])
        assert pd.Timestamp(detail["inner_valid_end"]) <= pd.Timestamp(detail["fit_end"])
        assert pd.Timestamp(detail["fit_end"]) < query.RENT_APPROVAL_DATE.min()
        assert not {"MONTHLY_RENT", "Id", "source_row"}.intersection(bundle.cats+bundle.nums)
        for column in bundle.cats:
            assert bundle.levels[column] == fit[column].fillna("missing").astype(str).unique().tolist()
        xquery = query.drop(columns="MONTHLY_RENT").copy()
        if manifest["grid"][name]["history"]:
            if split not in feature_snapshots:
                origin = fit.RENT_APPROVAL_DATE.dt.to_period("M").max()+1
                past = fit.loc[fit.RENT_APPROVAL_DATE.dt.to_period("M").ge(origin-12)]
                feature_snapshots[split] = snapshot(past, xquery)
            xquery[ALL_FEATURES] = feature_snapshots[split]
        prediction = bundle.predict(xquery)
        np.testing.assert_allclose(prediction, tables[split][name], atol=1e-8, rtol=0)
        with np.load(directory/"prediction.npz") as saved:
            np.testing.assert_array_equal(saved["source_row"], query.index)
            np.testing.assert_allclose(prediction, saved["prediction"], atol=1e-8, rtol=0)
        xquery["MONTHLY_RENT"], xquery["Id"] = -123456., 99999
        np.testing.assert_array_equal(prediction, bundle.predict(xquery))
        native_checks += 1
    assert sha(ROOT/"submissions/submission_rmse448_trend25_20261019.csv") == "e1f06bc525d0e048f04d7c86dcf069faf855fafdbd45d00438c69499ccbf5317"
    pd.DataFrame(diagnostic_rows).to_csv(output/"monthly_errors.csv", index=False)
    verification = dict(ok=True, score_checks=score_checks, selection_checks=len(expected_names),
        native_model_checks=native_checks, run_id=manifest["run_id"],
        best_pure=lock["best_pure"], best_new=lock["best_new"],
        previous_submission_unchanged=True, auditor_sha256=sha(Path(__file__)))
    dump(output/"output_validation.json", verification)
    print(json.dumps(verification, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT/"reports/experiments/xgboost_native_v1")
    inspect(parser.parse_args().output_dir.resolve())
