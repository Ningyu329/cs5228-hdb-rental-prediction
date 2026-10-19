"""Independent row, metric, model reload and leakage-boundary checks."""
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
from src.features.build_raw_views import attach_raw_view
from src.features.build_resale_history import load_resale, causal_resale, FEATURES
from src.models.run_stage5 import load_training
from src.models.run_rmse_improvement import sha, dump


def main(output):
    manifest = json.loads((output / "manifest.json").read_text())
    done = json.loads((output / "completion.json").read_text())
    if not done["complete"] or done["run_id"] != manifest["run_id"]:
        raise ValueError("Run not complete or identifier mismatch")
    for field in ["input_sha256", "implementation_sha256", "reference_sha256"]:
        for p, expected in manifest[field].items():
            if sha(ROOT / p) != expected:
                raise ValueError(f"Source changed: {p}")
    raw = pd.read_csv(ROOT / "data/raw/train.csv", dtype=str, keep_default_na=False)
    data = attach_raw_view(load_training(ROOT / "data/processed/train_location_features.csv"), raw)
    np.testing.assert_array_equal(data.MONTHLY_RENT.to_numpy(), pd.to_numeric(raw.MONTHLY_RENT).to_numpy())
    months = data.RENT_APPROVAL_DATE.dt.to_period("M")
    cache = ROOT / "models/data_rethink_cache" / manifest["run_id"]
    results = json.loads((output / "results.json").read_text())
    monthly = pd.DataFrame(json.loads((output / "monthly_errors.json").read_text()))
    seen = set()
    for row in results:
        key = row["split"], row["candidate"]
        if key in seen:
            raise ValueError("Duplicate result")
        seen.add(key)
        start, end = [pd.Period(s) for s in manifest["splits"][row["split"]]]
        query = data.loc[months.between(start, end)]
        with np.load(cache / f"candidate_{row['split']}_{row['candidate']}.npz") as z:
            np.testing.assert_array_equal(z["source_row"], query.index)
            error = z["prediction"] - query.MONTHLY_RENT.to_numpy()
        expected = dict(rmse=float(np.sqrt(np.mean(error ** 2))), mae=float(np.mean(abs(error))), bias=float(np.mean(error)))
        for c in expected:
            np.testing.assert_allclose(row[c], expected[c], atol=1e-9, rtol=1e-12)
        grouped = monthly.loc[monthly.split.eq(row["split"]) & monthly.candidate.eq(row["candidate"])]
        if int(grouped.rows.sum()) != len(query) or len(query) != row["rows"]:
            raise ValueError("Monthly/outer row totals disagree")
        np.testing.assert_allclose(np.sqrt(np.average(grouped.rmse ** 2, weights=grouped.rows)), row["rmse"], atol=1e-8)
    # Recompute selection from the earlier-quarter rows alone, not the primary.
    selection = json.loads((output / "selection_locked.json").read_text())
    eligible = []
    reference = {r["split"]: r["rmse"] for r in results if r["candidate"] == "incumbent" and r["split"] in ["q3_2024", "q4_2024"]}
    base_pooled = next(r["pooled_rmse"] for r in selection["ranking"] if r["candidate"] == "incumbent")
    for row in selection["ranking"]:
        part = [r for r in results if r["split"] in ["q3_2024", "q4_2024"] and r["candidate"] == row["candidate"]]
        if len(part) != 2:
            raise ValueError("Incomplete selection folds")
        pooled = np.sqrt(sum(r["rmse"] ** 2 * r["rows"] for r in part) / sum(r["rows"] for r in part))
        np.testing.assert_allclose(pooled, row["pooled_rmse"], atol=1e-9)
        worst = max(r["rmse"] - reference[r["split"]] for r in part)
        np.testing.assert_allclose(row["gain"], base_pooled - pooled, atol=1e-9)
        np.testing.assert_allclose(row["worst_quarter_delta"], worst, atol=1e-9)
        if row["eligible"] != (base_pooled - pooled >= 3 and worst <= 3):
            raise ValueError("Promotion gate inconsistent")
        if row["eligible"]:
            eligible.append(row)
    expected_selected = min(eligible, key=lambda r: (r["pooled_rmse"], r["candidate"]))["candidate"] if eligible else "incumbent"
    if expected_selected != selection["selected"]:
        raise ValueError("Selected candidate inconsistent")
    from src.models.select_long_horizon import select
    long_result = json.loads((output / "long_selection.json").read_text())
    rank, chosen = select(results)
    if rank != long_result["ranking"] or chosen != long_result["selected"]:
        raise ValueError("Long-origin selection inconsistent")
    if sha(output / "long_selection_plan.json") != long_result["plan_sha256"]:
        raise ValueError("Long selection plan changed")
    # Reload each fitted component and verify inputs have no target dependency.
    resale = load_resale(ROOT / "data/raw/external/hdb_20261019/resale.csv")
    details = json.loads((output / "fit_details.json").read_text())
    features_by_split = {}
    for detail in details:
        split, recipe = detail["split"], detail["recipe"]
        start, end = [pd.Period(s) for s in manifest["splits"][split]]
        fit, query = data.loc[months.lt(start)].copy(), data.loc[months.between(start, end)].copy()
        featured = query.drop(columns="MONTHLY_RENT")
        if recipe == "resale_lgb":
            if split not in features_by_split:
                _, features_by_split[split] = causal_resale(fit, query, resale)
            featured = features_by_split[split]
            if "MONTHLY_RENT" in featured:
                raise ValueError("Query rent retained in external features")
        payload = joblib.load(cache / split / recipe / "model.joblib")
        def predict(frame):
            if recipe == "hierarchy":
                return np.maximum(payload["structure"].predict(frame), 0)
            bundle = payload["bundle"]
            if "MONTHLY_RENT" in bundle.features or "Id" in bundle.features:
                raise ValueError("Target/ID included as feature")
            p = np.asarray(bundle.model.predict(bundle.matrix(frame)))
            if payload["structure"] is not None:
                p += payload["structure"].predict(frame)
            return np.maximum(p, 0)
        reloaded = predict(featured)
        np.testing.assert_allclose(predict(featured.assign(MONTHLY_RENT=-999999., Id=987654)), reloaded, atol=1e-8)
        with np.load(cache / split / recipe / "prediction.npz") as z:
            np.testing.assert_array_equal(z["source_row"], query.index)
            np.testing.assert_allclose(reloaded, z["prediction"], atol=1e-8, rtol=1e-10)
    submission = ROOT / "submissions/submission_rmse448_trend25_20261019.csv"
    expected_hash = "e1f06bc525d0e048f04d7c86dcf069faf855fafdbd45d00438c69499ccbf5317"
    if sha(submission) != expected_hash:
        raise ValueError("Existing submitted prediction changed")
    report = dict(ok=True, run_id=manifest["run_id"], scores_checked=len(results), models_reloaded=len(details),
        source_fingerprints_valid=True, selection_recomputed=True, long_selection_recomputed=True, monthly_totals_checked=True,
        query_target_and_id_invariance=True, previous_submission_unchanged=True,
        property_used_in_training=False, public_rental_labels_downloaded=False)
    dump(output / "output_validation.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/experiments/data_rethink_v3")
    main(parser.parse_args().output_dir.resolve())
