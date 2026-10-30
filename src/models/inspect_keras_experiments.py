"""Reconcile neural metrics, seed means, preprocessing cutoffs and saved models."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if __package__ in {None, ""}:
    sys.path.insert(0, str(ROOT))
from src.models.keras_rent import load_bundle, recipe_schema, numeric_values, release
from src.models.run_stage5 import load_training
from src.models.run_rmse_improvement import EARLIER, dump, sha
from src.models.run_keras_experiments import CONTROL, REFERENCE


def compare_metrics(row, actual, predicted):
    error = predicted - actual
    expected = [np.sqrt(np.mean(error ** 2)), np.mean(np.abs(error)), error.mean()]
    np.testing.assert_allclose([row[k] for k in ["rmse", "mae", "bias"]], expected, atol=1e-8, rtol=1e-12)


def inspect(output):
    manifest = json.loads((output / "experiment_manifest.json").read_text())
    complete = json.loads((output / "completion.json").read_text())
    lock = json.loads((output / "selection_locked.json").read_text())
    assert complete["completed"] and manifest["run_id"] == complete["run_id"] == lock["run_id"]
    assert not manifest["test_data_read"] and not manifest["submission_generated"] and not lock["primary_used"]
    for mapping in [manifest["implementation_sha256"], manifest["reference_sha256"],
                    {manifest["source"]: manifest["source_sha256"]}]:
        for path, expected in mapping.items():
            assert sha(ROOT / path) == expected, path
    data = load_training(ROOT / manifest["source"])
    dates = data.RENT_APPROVAL_DATE.dt.to_period("M")
    cache = ROOT / "models/keras_cache" / manifest["run_id"]
    results = pd.read_csv(output / "results.csv")
    seeds = pd.read_csv(output / "seed_results.csv")
    details = json.loads((output / "fit_details.json").read_text())
    assert not seeds.duplicated(["split", "component", "seed"]).any()
    assert len(details) == len(seeds) == complete["fitted_models"] == 30
    assert not results.duplicated(["split", "candidate"]).any()
    tables, queries = {}, {}
    for split in results.split.unique():
        start, end = [pd.Period(s) for s in manifest["splits"][split]]
        query = data.loc[dates.between(start, end)]
        queries[split] = query
        table = pd.read_csv(output / f"predictions_{split}.csv")
        tables[split] = table
        np.testing.assert_array_equal(query.index, table.source_row)
        np.testing.assert_array_equal(query.MONTHLY_RENT, table.actual)
        old = pd.read_csv(REFERENCE / f"predictions_{split}.csv")
        np.testing.assert_array_equal(old.source_row, table.source_row)
        np.testing.assert_allclose(old[CONTROL], table.incumbent, atol=1e-8, rtol=0)
    component_predictions = {}
    for detail in details:
        split, kind, seed = detail["split"], detail["component"], detail["seed"]
        assert seed in manifest["seeds"] and kind in manifest["grid"]
        query = queries[split]
        start = pd.Period(manifest["splits"][split][0])
        fit = data.loc[dates.lt(start)]
        assert pd.Timestamp(detail["inner_fit_end"]) < pd.Timestamp(detail["inner_valid_start"])
        assert pd.Timestamp(detail["inner_valid_end"]) == fit.RENT_APPROVAL_DATE.max()
        assert pd.Timestamp(detail["fit_end"]) < query.RENT_APPROVAL_DATE.min()
        history = detail["inner_history"]["history"]["val_scaled_mse"]
        assert detail["best_epoch"] == int(np.argmin(history)) + 1
        assert detail["refit_history"]["epochs_run"] == detail["best_epoch"]
        directory = cache / split / kind / f"seed_{seed}"
        for file, expected in detail["saved_sha256"].items():
            assert sha(directory / file) == expected
        with np.load(directory / "prediction.npz") as saved:
            np.testing.assert_array_equal(saved["source_row"], query.index)
            stored = saved["prediction"].copy()
        row = seeds.loc[seeds.split.eq(split) & seeds.component.eq(kind) & seeds.seed.eq(seed)].iloc[0]
        compare_metrics(row, query.MONTHLY_RENT.to_numpy(), stored)
        bundle = load_bundle(directory)
        cats, nums = recipe_schema(kind)
        assert bundle.schema["cats"] == cats and bundle.schema["nums"] == nums
        for col in cats:
            counts = fit[col].fillna("missing").astype(str).value_counts()
            assert bundle.schema["levels"][col] == sorted(counts.index[counts.ge(2)].tolist())
        x = numeric_values(fit, nums)
        medians = [np.median(x[np.isfinite(x[:, i]), i]) if np.isfinite(x[:, i]).any() else 0.
                   for i in range(len(nums))]
        np.testing.assert_allclose(bundle.schema["medians"], medians, atol=1e-10)
        filled = np.where(np.isfinite(x), x, medians)
        np.testing.assert_allclose(bundle.schema["means"], filled.mean(axis=0), atol=1e-10)
        std = filled.std(axis=0)
        np.testing.assert_allclose(bundle.schema["scales"], np.where(std > 1e-8, std, 1.), atol=1e-10)
        if kind == "trend":
            assert bundle.trend["cutoff"] == fit.RENT_APPROVAL_DATE.max().year * 12 + fit.RENT_APPROVAL_DATE.max().month
        prediction = bundle.predict(query)
        np.testing.assert_allclose(prediction, stored, atol=1e-5, rtol=0)
        np.testing.assert_allclose(bundle.predict(query.assign(MONTHLY_RENT=-123456., Id=12345)), stored, atol=1e-5, rtol=0)
        component_predictions.setdefault((split, kind), []).append((seed, stored))
        release(bundle)
        del bundle
        print(f"VERIFIED {split}/{kind}/{seed}", flush=True)
    monthly, seed_blends = [], []
    selected_kind, selected_percent = lock["best_new"].rsplit("_blend", 1)
    for (split, kind), values in component_predictions.items():
        assert sorted(seed for seed, _ in values) == manifest["seeds"]
        ensemble = np.mean([p for _, p in values], axis=0)
        table = tables[split]
        np.testing.assert_allclose(table[f"{kind}_blend100"], ensemble, atol=1e-8, rtol=0)
        if kind == selected_kind:
            weight = int(selected_percent) / 100
            for seed, pred in values:
                error = (1 - weight) * table.incumbent.to_numpy() + weight * pred - table.actual.to_numpy()
                seed_blends.append(dict(split=split, candidate=lock["best_new"], seed=seed, rows=len(error),
                    rmse=float(np.sqrt(np.mean(error ** 2))), mae=float(np.mean(abs(error))), bias=float(error.mean())))
    for row in results.to_dict("records"):
        table, query = tables[row["split"]], queries[row["split"]]
        pred = table[row["candidate"]].to_numpy()
        if row["candidate"] != "incumbent":
            kind, percent = row["candidate"].rsplit("_blend", 1)
            weight = int(percent) / 100
            expected = (1 - weight) * table.incumbent + weight * table[f"{kind}_blend100"]
            np.testing.assert_allclose(pred, expected, atol=1e-8, rtol=0)
        compare_metrics(row, table.actual.to_numpy(), pred)
        correlation = np.corrcoef(pred-table.actual, table.incumbent-table.actual)[0, 1]
        np.testing.assert_allclose(row["residual_correlation_with_incumbent"], correlation, atol=1e-12)
        for month in sorted(query.RENT_APPROVAL_DATE.dt.strftime("%Y-%m").unique()):
            mask = query.RENT_APPROVAL_DATE.dt.strftime("%Y-%m").eq(month).to_numpy()
            error = pred[mask] - table.actual.to_numpy()[mask]
            monthly.append(dict(split=row["split"], candidate=row["candidate"], month=month, rows=int(mask.sum()),
                rmse=float(np.sqrt(np.mean(error ** 2))), mae=float(np.mean(abs(error))), bias=float(error.mean())))
    selection = pd.read_csv(output / "selection.csv")
    eligible = []
    earlier = results.loc[results.split.isin(EARLIER)]
    for row in selection.to_dict("records"):
        group = earlier.loc[earlier.candidate.eq(row["candidate"])]
        assert set(group.split) == set(EARLIER) and len(group) == 2
        pooled = np.sqrt(np.average(group.rmse ** 2, weights=group.rows))
        np.testing.assert_allclose(row["pooled_rmse"], pooled, atol=1e-9)
        base = float(selection.loc[selection.candidate.eq("incumbent"), "pooled_rmse"].iloc[0])
        worst = max(r.rmse - float(earlier.loc[earlier.split.eq(r.split) & earlier.candidate.eq("incumbent"), "rmse"].iloc[0])
                    for r in group.itertuples())
        np.testing.assert_allclose([row["gain"], row["worst_quarter_delta"]], [base-pooled, worst], atol=1e-9)
        assert row["eligible"] == (base - pooled >= 1. and worst <= 1.)
        if row["eligible"]:
            eligible.append((pooled, row["candidate"]))
    expected_selection = min(eligible)[1] if eligible else "incumbent"
    assert lock["selected"] == expected_selection
    ordered = selection.sort_values(["pooled_rmse", "candidate"])
    assert lock["best_pure"] == ordered.loc[ordered.candidate.str.endswith("_blend100")].iloc[0].candidate
    assert lock["best_new"] == ordered.loc[~ordered.candidate.eq("incumbent")].iloc[0].candidate
    pb = json.loads((ROOT / "reports/submissions/personal_best.json").read_text())
    for key in ["personal_best", "latest_submission"]:
        assert sha(ROOT / pb[key]["file"]) == pb[key]["sha256"]
    pd.DataFrame(monthly).to_csv(output / "monthly_errors.csv", index=False)
    pd.DataFrame(seed_blends).to_csv(output / "seed_blends.csv", index=False)
    validation = dict(ok=True, run_id=manifest["run_id"], scores_checked=len(results), seed_scores_checked=len(seeds),
        models_reloaded=len(details), seed_ensembles_checked=len(component_predictions),
        preprocessing_fit_only=True, source_fingerprints_valid=True, query_target_and_id_invariance=True,
        selection_excludes_primary=True, original_submissions_unchanged=True, production_replaced=False,
        test_data_read=False, inspector_sha256=sha(Path(__file__)))
    dump(output / "output_validation.json", validation)
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/experiments/keras_v1")
    inspect(parser.parse_args().output_dir.resolve())
