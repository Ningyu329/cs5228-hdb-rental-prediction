"""Independent reconciliation of TabM outputs, source rows, seeds and models."""
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
from src.models.tabm_rent import load_bundle, fit_tabm_schema, tensors, torch_api
from src.features.build_temporal_features import fit_trend, trend_values
from src.models.run_stage5 import load_training
from src.models.run_rmse_improvement import EARLIER, dump, sha, signature
from src.models.run_keras_experiments import CONTROL, REFERENCE


def metric_check(row, actual, prediction):
    error = prediction - actual
    values = [np.sqrt(np.mean(error ** 2)), np.mean(abs(error)), np.mean(error)]
    np.testing.assert_allclose([row[k] for k in ["rmse", "mae", "bias"]], values, atol=1e-8, rtol=1e-12)


def inspect(output, screening_only=False):
    manifest = json.loads((output / "experiment_manifest.json").read_text())
    assert signature({k: v for k, v in manifest.items() if k != "run_id"}) == manifest["run_id"]
    complete = json.loads((output / "completion.json").read_text())
    lock = json.loads((output / "selection_locked.json").read_text())
    assert manifest["run_id"] == complete["run_id"] == lock["run_id"]
    if screening_only:
        assert lock["selected"] == "incumbent" and not complete["completed"]
    else:
        assert complete["completed"]
    assert not manifest["test_data_read"] and not manifest["submission_generated"] and not lock["primary_used"]
    for mapping in [manifest["implementation_sha256"], manifest["reference_sha256"],
                    {manifest["source"]: manifest["source_sha256"]}]:
        for path, expected in mapping.items():
            assert sha(ROOT / path) == expected, path
    data = load_training(ROOT / manifest["source"])
    months = data.RENT_APPROVAL_DATE.dt.to_period("M")
    cache = ROOT / "models/tabm_cache" / manifest["run_id"]
    results = pd.read_csv(output / "results.csv")
    diagnostic_reference_rows = 0
    if screening_only:
        # A normal interrupt immediately after the lock can occur just after
        # the already-known incumbent diagnostic row is copied. It is not a
        # neural diagnostic score and cannot affect the frozen selection.
        references = results.loc[~results.split.isin(EARLIER)]
        assert references.candidate.eq("incumbent").all()
        diagnostic_reference_rows = len(references)
        results = results.loc[results.split.isin(EARLIER)].copy()
    seeds = pd.read_csv(output / "seed_results.csv")
    details = json.loads((output / "fit_details.json").read_text())
    assert not seeds.duplicated(["split", "component", "seed"]).any()
    expected_models = 12 if screening_only else 30
    assert len(details) == len(seeds) == expected_models
    if not screening_only:
        assert complete["fitted_models"] == expected_models
    else:
        assert set(results.split) == set(EARLIER) and len(results) == 18
    assert not results.duplicated(["split", "candidate"]).any()
    tables, queries, fitting = {}, {}, {}
    for split in results.split.unique():
        start, end = [pd.Period(s) for s in manifest["splits"][split]]
        query = data.loc[months.between(start, end)]
        fitting[split], queries[split] = data.loc[months.lt(start)], query
        table = pd.read_csv(output / f"predictions_{split}.csv")
        tables[split] = table
        np.testing.assert_array_equal(query.index, table.source_row)
        np.testing.assert_array_equal(query.MONTHLY_RENT, table.actual)
        old = pd.read_csv(REFERENCE / f"predictions_{split}.csv")
        np.testing.assert_array_equal(old.source_row, table.source_row)
        np.testing.assert_allclose(old[CONTROL], table.incumbent, atol=1e-8, rtol=0)
    component_predictions, reload_differences, reused = {}, [], 0
    for detail in details:
        split, kind, seed = detail["split"], detail["component"], detail["seed"]
        assert seed in manifest["seeds"] and kind in manifest["grid"]
        fit, query = fitting[split], queries[split]
        assert detail["fit_rows"] == len(fit) and detail["query_rows"] == len(query)
        assert pd.Timestamp(detail["inner_fit_end"]) < pd.Timestamp(detail["inner_valid_start"])
        assert pd.Timestamp(detail["inner_valid_end"]) == fit.RENT_APPROVAL_DATE.max()
        assert pd.Timestamp(detail["fit_end"]) < query.RENT_APPROVAL_DATE.min()
        inner_history = detail["inner_history"]["history"]
        assert detail["best_epoch"] == int(np.argmin([r["validation_scaled_mse"] for r in inner_history])) + 1
        assert detail["refit_history"]["epochs_run"] == detail["best_epoch"]
        directory = cache / split / kind / f"seed_{seed}"
        for file, expected in detail["saved_sha256"].items():
            assert sha(directory / file) == expected
        if "reused_training_from" in detail:
            assert split == "long_6m"
            np.testing.assert_array_equal(fit.index, fitting["q3_2024"].index)
            assert Path(detail["reused_training_from"]) == (cache / "q3_2024" / kind / f"seed_{seed}").relative_to(ROOT)
            for file in ["model.pt", "schema.json"]:
                assert sha(directory / file) == sha(ROOT / detail["reused_training_from"] / file)
            reused += 1
        with np.load(directory / "prediction.npz") as stored:
            np.testing.assert_array_equal(stored["source_row"], query.index)
            prediction = stored["prediction"].copy()
        row = seeds.loc[seeds.split.eq(split) & seeds.component.eq(kind) & seeds.seed.eq(seed)].iloc[0]
        metric_check(row, query.MONTHLY_RENT.to_numpy(), prediction)
        bundle = load_bundle(directory)
        assert bundle.spec == manifest["grid"][kind]
        assert bundle.schema == fit_tabm_schema(fit, kind, bundle.spec["n_bins"])
        assert not {"MONTHLY_RENT", "Id", "source_row"}.intersection(bundle.schema["cats"] + bundle.schema["nums"])
        expected_trend = fit_trend(fit, "damped24") if kind == "trend" else None
        assert bundle.trend == expected_trend
        labels = fit.MONTHLY_RENT.to_numpy(dtype=float)
        if bundle.trend is not None:
            labels = labels - trend_values(bundle.trend, fit)
        np.testing.assert_allclose([bundle.center, bundle.scale], [labels.mean(), max(labels.std(), 1.)], atol=1e-10)
        reloaded = bundle.predict(query)
        maximum = float(np.max(abs(reloaded-prediction)))
        np.testing.assert_allclose(reloaded, prediction, atol=manifest["reload_tolerance_sgd"], rtol=0)
        np.testing.assert_allclose(bundle.predict(query.assign(MONTHLY_RENT=-123456., Id=12345)), reloaded,
                                   atol=manifest["reload_tolerance_sgd"], rtol=0)
        # Explicitly check that inference averages distinct branch outputs.
        torch = torch_api(bundle.device)
        with torch.inference_mode():
            branches = bundle.model(*tensors(bundle.schema, query.iloc[:16], bundle.device))
            assert branches.shape == (min(16, len(query)), bundle.spec["k"])
            mean = branches.mean(1).cpu().numpy().astype(float) * bundle.scale + bundle.center
        if bundle.trend is not None:
            mean += trend_values(bundle.trend, query.iloc[:16])
        np.testing.assert_allclose(np.maximum(mean, 0.), reloaded[:16], atol=.02, rtol=0)
        reload_differences.append(maximum)
        component_predictions.setdefault((split, kind), []).append((seed, prediction))
        del bundle
        print(f"VERIFIED {split}/{kind}/{seed}: reload maximum {maximum:.6f} SGD", flush=True)
    assert reused == (0 if screening_only else 6)
    for (split, kind), values in component_predictions.items():
        assert sorted(seed for seed, _ in values) == manifest["seeds"]
        ensemble = np.mean([p for _, p in values], axis=0)
        np.testing.assert_allclose(tables[split][f"{kind}_blend100"], ensemble, atol=1e-8, rtol=0)
    monthly, seed_blends, primary_groups = [], [], []
    selected_kind, selected_percent = lock["best_new"].rsplit("_blend", 1)
    for (split, kind), values in component_predictions.items():
        if kind == selected_kind:
            table = tables[split]
            weight = int(selected_percent) / 100
            for seed, pred in values:
                error = (1-weight)*table.incumbent.to_numpy()+weight*pred-table.actual.to_numpy()
                seed_blends.append(dict(split=split, candidate=lock["best_new"], seed=seed, rows=len(error),
                    rmse=float(np.sqrt(np.mean(error**2))), mae=float(np.mean(abs(error))), bias=float(error.mean())))
    for row in results.to_dict("records"):
        split, candidate = row["split"], row["candidate"]
        table = tables[split]
        prediction = table[candidate].to_numpy()
        metric_check(row, table.actual.to_numpy(), prediction)
        if candidate != "incumbent":
            kind, percentage = candidate.rsplit("_blend", 1)
            weight = int(percentage)/100
            np.testing.assert_allclose(prediction, (1-weight)*table.incumbent+weight*table[f"{kind}_blend100"], atol=1e-8)
        correlation = np.corrcoef(prediction-table.actual, table.incumbent-table.actual)[0, 1]
        np.testing.assert_allclose(correlation, row["residual_correlation_with_incumbent"], atol=1e-12)
        if split == "primary_2025":
            # Retrospective diagnosis only: these true-rent groups are not
            # available as test-time routing features or used for selection.
            actual = table.actual.to_numpy()
            for name, mask in [("actual_rent_le_2000", actual <= 2000), ("actual_rent_gt_2000", actual > 2000)]:
                error = prediction[mask] - actual[mask]
                primary_groups.append(dict(candidate=candidate, diagnostic_group=name, rows=int(mask.sum()),
                    rmse=float(np.sqrt(np.mean(error**2))), mae=float(np.mean(abs(error))),
                    bias=float(error.mean()), sse=float(np.sum(error**2)), retrospective_only=True))
        # Compute dates once per candidate, not once per month.
        query_months = queries[split].RENT_APPROVAL_DATE.dt.strftime("%Y-%m").to_numpy()
        for month in sorted(set(query_months)):
            mask = query_months == month
            error = prediction[mask] - table.actual.to_numpy()[mask]
            monthly.append(dict(split=split, candidate=candidate, month=month, rows=int(mask.sum()),
                rmse=float(np.sqrt(np.mean(error**2))), mae=float(np.mean(abs(error))), bias=float(error.mean())))
    selection = pd.read_csv(output / "selection.csv")
    earlier = results.loc[results.split.isin(EARLIER)]
    control = earlier.loc[earlier.candidate.eq("incumbent")].set_index("split")
    base = float(np.sqrt(np.average(control.rmse**2, weights=control.rows)))
    eligible = []
    for row in selection.to_dict("records"):
        group = earlier.loc[earlier.candidate.eq(row["candidate"])]
        assert set(group.split) == set(EARLIER) and len(group) == 2
        pooled = float(np.sqrt(np.average(group.rmse**2, weights=group.rows)))
        worst = max(r.rmse-control.loc[r.split, "rmse"] for r in group.itertuples())
        np.testing.assert_allclose([row["pooled_rmse"], row["gain"], row["worst_quarter_delta"]],
                                   [pooled, base-pooled, worst], atol=1e-9)
        assert row["eligible"] == (base-pooled >= 1. and worst <= 1.)
        if row["eligible"]:
            eligible.append((pooled, row["candidate"]))
    assert lock["selected"] == (min(eligible)[1] if eligible else "incumbent")
    ordered = selection.sort_values(["pooled_rmse", "candidate"])
    assert lock["best_pure"] == ordered.loc[ordered.candidate.str.endswith("_blend100")].iloc[0].candidate
    assert lock["best_new"] == ordered.loc[~ordered.candidate.eq("incumbent")].iloc[0].candidate
    pb = json.loads((ROOT / "reports/submissions/personal_best.json").read_text())
    for key in ["personal_best", "latest_submission"]:
        assert sha(ROOT / pb[key]["file"]) == pb[key]["sha256"]
    pd.DataFrame(monthly).to_csv(output / "monthly_errors.csv", index=False)
    pd.DataFrame(seed_blends).to_csv(output / "seed_blends.csv", index=False)
    if primary_groups:
        pd.DataFrame(primary_groups).to_csv(output / "primary_diagnostic_groups.csv", index=False)
    result = dict(ok=True, run_id=manifest["run_id"], scores_checked=len(results), seed_scores_checked=len(seeds),
        scope="earlier_quarter_screening" if screening_only else "full_diagnostics",
        screening_complete=True, full_diagnostics_completed=not screening_only,
        known_incumbent_diagnostic_rows=diagnostic_reference_rows,
        model_records_reloaded=len(details), freshly_trained_models=len(details)-reused,
        shared_origin_reuses_verified=reused, seed_ensembles_checked=len(component_predictions),
        maximum_reload_difference_sgd=max(reload_differences), preprocessing_fit_only=True,
        source_fingerprints_valid=True, separate_branches_verified=True, query_target_and_id_invariance=True,
        selection_excludes_primary=True, original_submissions_unchanged=True,
        production_replaced=False, test_data_read=False, inspector_sha256=sha(Path(__file__)))
    dump(output / "output_validation.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/experiments/tabm_v1")
    parser.add_argument("--screening-only", action="store_true")
    args = parser.parse_args()
    inspect(args.output_dir.resolve(), args.screening_only)
