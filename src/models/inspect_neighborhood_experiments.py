"""Independently recompute scores, selection and saved-model predictions."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if __package__ in {None, ""}:
    sys.path.insert(0, str(ROOT))

from src.features.build_neighborhood_features import ALL_FEATURES, DISTRIBUTION, SPATIAL, snapshot
from src.models.run_stage5 import load_training
from src.models.run_rmse_improvement import SPLITS, EARLIER, dump, sha


def inspect(output):
    manifest = json.loads((output / "experiment_manifest.json").read_text())
    complete = json.loads((output / "completion.json").read_text())
    lock = json.loads((output / "selection_locked.json").read_text())
    identity = {k: v for k, v in manifest.items() if k != "run_id"}
    assert hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20] == manifest["run_id"]
    assert complete["completed"] and manifest["run_id"] == lock["run_id"] == complete["run_id"]
    assert not manifest["test_data_read"] and not manifest["submission_generated"]
    assert not lock["primary_used"] and lock["selection_splits"] == EARLIER
    assert manifest["source_sha256"] == sha(ROOT / manifest["source"])
    for section in ["implementation_sha256", "reference_sha256"]:
        for path, expected in manifest[section].items():
            assert sha(ROOT / path) == expected, path
    data = load_training(ROOT / manifest["source"])
    months = data.RENT_APPROVAL_DATE.dt.to_period("M")
    results = pd.read_csv(output / "results.csv")
    assert not results.duplicated(["split", "candidate"]).any()
    frames, predictions = {}, {}
    checks = 0
    for split in results.split.unique():
        start, end = [pd.Period(s) for s in SPLITS[split]]
        fit, query = data.loc[months.lt(start)], data.loc[months.between(start, end)]
        frames[split] = fit, query
        table = pd.read_csv(output / f"predictions_{split}.csv")
        predictions[split] = table
        np.testing.assert_array_equal(table.source_row, query.index)
        np.testing.assert_array_equal(table.actual, query.MONTHLY_RENT)
        old = pd.read_csv(ROOT / f"reports/experiments/rmse_improvement_v1/predictions_{split}.csv")
        np.testing.assert_allclose(table.incumbent, old.trend_damped24_weight25, atol=1e-8, rtol=0)
        for row in results.loc[results.split.eq(split)].itertuples():
            pred = table[row.candidate].to_numpy()
            error = pred - query.MONTHLY_RENT.to_numpy()
            np.testing.assert_allclose([np.sqrt(np.mean(error**2)), np.mean(np.abs(error)), np.mean(error)],
                                       [row.rmse, row.mae, row.bias], atol=1e-8, rtol=0)
            assert len(query) == row.rows
            if row.candidate != "incumbent":
                component, percent = row.candidate.rsplit("_blend", 1)
                weight = int(percent) / 100
                expected = (1-weight)*table.incumbent + weight*table[component]
                np.testing.assert_allclose(pred, expected, atol=1e-8, rtol=0)
            checks += 1
    # Recompute pooled SSE directly from errors, not from rounded result RMSE.
    earlier = results.loc[results.split.isin(EARLIER)]
    selection = pd.read_csv(output / "selection.csv").set_index("candidate")
    base_errors = np.concatenate([predictions[s].incumbent-predictions[s].actual for s in EARLIER])
    base = np.sqrt(np.mean(base_errors**2))
    independent = []
    names = set(earlier.candidate)
    assert set(selection.index) == names
    for name in names:
        assert set(earlier.loc[earlier.candidate.eq(name)].split) == set(EARLIER)
        error = np.concatenate([predictions[s][name]-predictions[s].actual for s in EARLIER])
        pooled = np.sqrt(np.mean(error**2))
        worst = max(np.sqrt(np.mean((predictions[s][name]-predictions[s].actual)**2)) -
                    np.sqrt(np.mean((predictions[s].incumbent-predictions[s].actual)**2)) for s in EARLIER)
        eligible = base - pooled >= 1. and worst <= 1.
        row = selection.loc[name]
        np.testing.assert_allclose([row.pooled_rmse, row.gain, row.worst_quarter_delta],
                                   [pooled, base-pooled, worst], atol=1e-8, rtol=0)
        assert bool(row.eligible) == eligible
        independent.append((pooled, name, eligible))
    independent.sort()
    accepted = [name for _, name, eligible in independent if eligible]
    selected = accepted[0] if accepted else "incumbent"
    challenger = next(name for _, name, _ in independent if name != "incumbent")
    assert (selected, challenger) == (lock["selected"], lock["challenger"])
    diagnostic_rows = []
    for split, table in predictions.items():
        _, query = frames[split]
        group_months = query.RENT_APPROVAL_DATE.dt.strftime("%Y-%m").to_numpy()
        groups = {f"month_{month}": group_months == month for month in sorted(set(group_months))}
        groups.update(actual_rent_le_2000=table.actual.to_numpy() <= 2000,
                      actual_rent_gt_2000=table.actual.to_numpy() > 2000)
        for name, mask in groups.items():
            for candidate in dict.fromkeys(["incumbent", selected, challenger]):
                if not mask.any():
                    continue
                error = (table[candidate]-table.actual).to_numpy()[mask]
                diagnostic_rows.append(dict(split=split, group=name, candidate=candidate, rows=int(mask.sum()),
                    rmse=float(np.sqrt(np.mean(error**2))), mae=float(np.mean(np.abs(error))), bias=float(error.mean())))
    pd.DataFrame(diagnostic_rows).to_csv(output / "diagnostic_groups.csv", index=False)
    # Reload every native model. Rebuild forecast features from a direct past
    # snapshot rather than reading the runner's feature cache.
    details = json.loads((output / "fit_details.json").read_text())
    assert len(details) == len({(d["split"], d["component"]) for d in details})
    native_checks = 0
    importance_rows = []
    feature_snapshots = {}
    for detail in details:
        split, name = detail["split"], detail["component"]
        fit, query = frames[split]
        path = ROOT / "models/neighborhood_cache" / manifest["run_id"] / split / name
        payload = joblib.load(path / "model.joblib")
        assert payload["detail"] == detail and payload["spec"] == manifest["grid"][name]
        spec, bundle = payload["spec"], payload["bundle"]
        assert not {"MONTHLY_RENT", "Id", "source_row"}.intersection(bundle.features)
        assert pd.Timestamp(detail["fit_end"]) < query.RENT_APPROVAL_DATE.min()
        assert pd.Timestamp(detail["inner_fit_end"]) < pd.Timestamp(detail["inner_valid_start"])
        assert pd.Timestamp(detail["inner_valid_end"]) <= pd.Timestamp(detail["fit_end"])
        assert 1 <= detail["iterations"] <= spec["cap"]
        xquery = query.drop(columns="MONTHLY_RENT").copy()
        if spec["features"]:
            if split not in feature_snapshots:
                origin = fit.RENT_APPROVAL_DATE.dt.to_period("M").max() + 1
                past = fit.loc[fit.RENT_APPROVAL_DATE.dt.to_period("M").ge(origin-12)]
                feature_snapshots[split] = snapshot(past, xquery)
            xquery[ALL_FEATURES] = feature_snapshots[split]
        pred = np.asarray(bundle.model.predict(bundle.matrix(xquery)), dtype=float)
        if spec["unit"]:
            pred *= xquery.FLOOR_AREA_SQM.to_numpy()
        pred = np.maximum(pred, 0)
        np.testing.assert_allclose(pred, predictions[split][name], atol=1e-8, rtol=0)
        # Labels inserted into the query cannot enter the model matrix.
        first_matrix = bundle.matrix(xquery)
        xquery["MONTHLY_RENT"] = -123456.
        pd.testing.assert_frame_equal(first_matrix, bundle.matrix(xquery))
        if bundle.family == "lightgbm":
            importance = bundle.model.booster_.feature_importance(importance_type="gain")
            importance_kind = "training_split_gain"
        else:
            importance = np.asarray(bundle.model.feature_importances_, dtype=float)
            importance_kind = "prediction_values_change"
        for feature, value in zip(bundle.features, importance):
            family = "distribution" if feature in DISTRIBUTION else "spatial" if feature in SPATIAL else "housing"
            importance_rows.append(dict(split=split, component=name, feature=feature, group=family,
                importance_kind=importance_kind, importance=float(value),
                share=float(value / importance.sum()) if importance.sum() else 0.))
        native_checks += 1
    pd.DataFrame(importance_rows).to_csv(output / "feature_importance.csv", index=False)
    submission = ROOT / "submissions/submission_rmse448_trend25_20261019.csv"
    assert sha(submission) == "e1f06bc525d0e048f04d7c86dcf069faf855fafdbd45d00438c69499ccbf5317"
    coverage = {split: dict(rows=len(frame),
        exact_at_least_1=float(frame.PAST_EXACT_COUNT.ge(1).mean()),
        exact_at_least_5=float(frame.PAST_EXACT_COUNT.ge(5).mean()),
        block_at_least_1=float(frame.PAST_BLOCK_COUNT.ge(1).mean()),
        block_at_least_5=float(frame.PAST_BLOCK_COUNT.ge(5).mean()),
        median_16th_block_distance_km=float(frame.NEAR_16_FARTHEST_KM.median()),
        median_64th_block_distance_km=float(frame.NEAR_64_FARTHEST_KM.median()))
        for split, frame in feature_snapshots.items()}
    result = dict(ok=True, score_checks=checks, selection_checks=len(names), native_model_checks=native_checks,
        source_hashes_verified=True, previous_submission_unchanged=True, selected=selected,
        challenger=challenger, history_coverage=coverage,
        auditor_sha256=sha(Path(__file__)), run_id=manifest["run_id"])
    dump(output / "output_validation.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/experiments/neighborhood_v1")
    inspect(parser.parse_args().output_dir.resolve())
