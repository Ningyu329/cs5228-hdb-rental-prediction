"""Refit the locked RMSE experiment on all labels and export one Kaggle CSV.

This does not submit to Kaggle or change the selected recipe. Early stopping
uses only the final three labelled training months, followed by full refitting.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if __package__ in {None, ""}:
    sys.path.insert(0, str(ROOT))

from src.data.clean_data import read_csv, clean_main_rows
from src.features.build_stage5_features import add_housing_features
from src.models.run_rmse_improvement import Experiment, Spec, dump, sha, signature
from src.models.run_stage5 import feature_schema
from src.models.train_baseline import TARGET


def checked_test(path: Path, raw: Path) -> pd.DataFrame:
    """Compare every cleaned input cell with the original test in row order."""
    frame = pd.read_csv(path)
    if TARGET in frame:
        raise ValueError("Test must not contain rental labels")
    rows, _ = read_csv(raw)
    cleaned, fields = clean_main_rows(rows, has_target=False)
    expected = pd.DataFrame(cleaned, columns=fields)
    if len(frame) != len(expected) or len(frame) != 50000:
        raise ValueError("Unexpected test row count")
    for column in fields:
        if column in {"TOWN", "BLOCK", "STREET", "FLAT_TYPE", "FLAT_MODEL", "FURNISHED", "RENT_APPROVAL_DATE"}:
            np.testing.assert_array_equal(frame[column].fillna("").astype(str), expected[column])
        else:
            np.testing.assert_allclose(pd.to_numeric(frame[column]),
                pd.to_numeric(expected[column].replace("", np.nan)), rtol=0, atol=1e-9, equal_nan=True)
    frame = add_housing_features(frame)
    cats, nums = feature_schema("housing")
    for column in cats:
        frame[column] = frame[column].fillna("missing").astype(str)
    for column in nums:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    frame["RENT_APPROVAL_DATE"] = pd.to_datetime(frame["RENT_APPROVAL_DATE"], format="ISO8601", errors="raise")
    return frame


def export(source_dir: Path, output: Path, report_dir: Path, threads: int = 4) -> dict:
    source_dir, output, report_dir = source_dir.resolve(), output.resolve(), report_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing submission: {output}")
    old = json.loads((source_dir / "experiment_manifest.json").read_text())
    lock_path = source_dir / "selection_locked.json"
    lock = json.loads(lock_path.read_text())
    if not old.get("complete") or lock["selected"] != old["selected"]:
        raise ValueError("Experiment must be complete and selected recipe consistent")
    for name, digest in old["implementation_sha256"].items():
        if sha(ROOT / name) != digest:
            raise ValueError(f"Selected implementation has changed: {name}")
    train_path = ROOT / old["source"]
    if sha(train_path) != old["source_sha256"]:
        raise ValueError("Training features have changed")
    test_path = ROOT / "data/processed/test_location_features.csv"
    raw_path = ROOT / "data/raw/test.csv"
    query = checked_test(test_path, raw_path)
    experiment = Experiment(train_path, report_dir, threads)
    code = sorted((ROOT / "src").rglob("*.py"))
    identity = dict(training_sha256=sha(train_path), test_sha256=sha(test_path),
        raw_test_sha256=sha(raw_path), lock_sha256=sha(lock_path),
        implementation_sha256={str(p.relative_to(ROOT)): sha(p) for p in code},
        environment=experiment.manifest["environment"], threads=threads)
    # Test hashes and all transitive source files are part of this export cache.
    experiment.cache = ROOT / "models/rmse_submission_cache" / signature(identity)
    experiment.cache.mkdir(parents=True, exist_ok=True)
    experiment.splits["competition_test"] = (experiment.train, query)
    experiment.specs = {name: Spec(**lock["specs"][name]) for name in lock["weights"]}
    experiment.register(lock["selected"], lock["weights"])
    prediction = experiment.predict("competition_test", lock["selected"])
    if not np.isfinite(prediction).all() or (prediction < 0).any():
        raise ValueError("Invalid predictions")
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"Id": np.arange(len(query)), "Predicted": prediction}).to_csv(output, index=False)
    saved = pd.read_csv(output)
    if list(saved) != ["Id", "Predicted"] or len(saved) != len(query):
        raise ValueError("Invalid saved submission shape")
    np.testing.assert_array_equal(saved.Id, np.arange(len(query)))
    np.testing.assert_allclose(saved.Predicted, prediction, rtol=0, atol=1e-9)
    audit = dict(**identity, selected=lock["selected"], weights=lock["weights"],
        train_rows=len(experiment.train), test_rows=len(query),
        fit_end=f"{experiment.train.RENT_APPROVAL_DATE.max():%Y-%m}",
        test_start=f"{query.RENT_APPROVAL_DATE.min():%Y-%m}",
        test_end=f"{query.RENT_APPROVAL_DATE.max():%Y-%m}",
        test_order_verified_against_raw=True, no_test_labels=True,
        output=str(output), output_sha256=sha(output),
        prediction_summary={key: float(value) for key, value in saved.Predicted.describe().items()},
        cache=str(experiment.cache), submission_generated=True, submitted_to_kaggle=False)
    dump(report_dir / "submission_manifest.json", audit)
    experiment.manifest.update(test_data_read=True, submission_generated=True,
        selected=lock["selected"], export_cache=str(experiment.cache))
    dump(report_dir / "experiment_manifest.json", experiment.manifest)
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=ROOT / "reports/experiments/rmse_improvement_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "submissions/submission_rmse448_trend25_20261019.csv")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports/submissions/rmse448_20261019")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    export(args.source_dir, args.output, args.report_dir, args.threads)
