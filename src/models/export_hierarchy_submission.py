"""Export one explicitly user-authorized challenger; never overwrite the PB CSV."""
from __future__ import annotations

import json
from pathlib import Path
import platform
import sys
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if __package__ in {None, ""}:
    sys.path.insert(0, str(ROOT))
from src.models.hierarchical_rent import HierarchicalRent
from src.models.run_stage5 import load_training
from src.models.export_rmse_submission import checked_test
from src.models.run_rmse_improvement import dump, sha, signature


def main():
    experiment = ROOT / "reports/experiments/data_rethink_v3"
    manifest = json.loads((experiment / "manifest.json").read_text())
    complete = json.loads((experiment / "completion.json").read_text())
    validation = json.loads((experiment / "output_validation.json").read_text())
    if not complete["complete"] or not validation["ok"] or complete["run_id"] != manifest["run_id"]:
        raise ValueError("Completed, independently validated experiment required")
    for group in ["input_sha256", "implementation_sha256", "reference_sha256"]:
        for path, digest in manifest[group].items():
            if sha(ROOT / path) != digest:
                raise ValueError(f"Experiment input/code changed: {path}")
    ranked = json.loads((experiment / "selection_locked.json").read_text())["ranking"]
    new = sorted([r for r in ranked if r["candidate"] != "incumbent"], key=lambda r: (r["pooled_rmse"], r["candidate"]))[0]
    if new["candidate"] != "hierarchy_blend25":
        raise ValueError("Chosen candidate is not the best new earlier-quarter recipe")
    base_path = ROOT / "submissions/submission_rmse448_trend25_20261019.csv"
    base_manifest = json.loads((ROOT / "reports/submissions/rmse448_20261019/submission_manifest.json").read_text())
    if sha(base_path) != base_manifest["output_sha256"]:
        raise ValueError("Old submitted predictions changed")
    train_path = ROOT / "data/processed/train_location_features.csv"
    test_path = ROOT / "data/processed/test_location_features.csv"
    raw_path = ROOT / "data/raw/test.csv"
    if sha(test_path) != base_manifest["test_sha256"] or sha(raw_path) != base_manifest["raw_test_sha256"]:
        raise ValueError("Test data differs from the verified previous export")
    output = ROOT / "submissions/submission_hierarchy25_20261019.csv"
    report = ROOT / "reports/submissions/hierarchy25_20261019"
    if output.exists() or (report / "submission_manifest.json").exists():
        raise FileExistsError("Refusing to overwrite an existing export or receipt")
    fit = load_training(train_path)
    query = checked_test(test_path, raw_path)
    baseline = pd.read_csv(base_path)
    if list(baseline) != ["Id", "Predicted"] or len(baseline) != len(query):
        raise ValueError("Baseline submission schema/length mismatch")
    np.testing.assert_array_equal(baseline.Id, np.arange(50000))
    if not np.isfinite(baseline.Predicted).all():
        raise ValueError("Nonfinite baseline predictions")
    import sklearn, scipy
    identity = dict(train_sha256=sha(train_path), test_sha256=sha(test_path), raw_test_sha256=sha(raw_path),
        base_sha256=sha(base_path), selection_sha256=sha(experiment / "selection_locked.json"),
        implementation_sha256={str(p.relative_to(ROOT)): sha(p) for p in [Path(__file__),
            ROOT / "src/models/hierarchical_rent.py", ROOT / "src/models/export_rmse_submission.py",
            ROOT / "src/models/run_stage5.py", ROOT / "src/features/build_stage5_features.py", ROOT / "src/data/clean_data.py"]},
        environment=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                         sklearn=sklearn.__version__, scipy=scipy.__version__),
        weights=dict(previous_submission=.75, hierarchy=.25), hierarchy=dict(alpha=30., damping_months=6.))
    cache = ROOT / "models/hierarchy_submission_cache" / signature(identity)
    cache.mkdir(parents=True, exist_ok=True)
    structure = HierarchicalRent().fit(fit)
    model_path = cache / "hierarchy.joblib"
    joblib.dump(structure, model_path)
    predictions = np.maximum(structure.predict(query), 0.)
    reloaded = joblib.load(model_path)
    np.testing.assert_allclose(np.maximum(reloaded.predict(query.assign(MONTHLY_RENT=-999999, Id=-1)), 0),
                               predictions, rtol=1e-10, atol=1e-8)
    combined = .75 * baseline.Predicted.to_numpy() + .25 * predictions
    if not np.isfinite(combined).all() or (combined < 0).any():
        raise ValueError("Invalid submission values")
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(dict(Id=np.arange(len(query)), Predicted=combined)).to_csv(output, index=False)
    saved = pd.read_csv(output)
    np.testing.assert_array_equal(saved.Id, np.arange(50000))
    np.testing.assert_allclose(saved.Predicted, combined, rtol=0, atol=1e-9)
    if list(saved) != ["Id", "Predicted"] or saved.isna().any().any():
        raise ValueError("Saved submission invalid")
    audit = dict(identity, candidate=new["candidate"], train_rows=len(fit), test_rows=len(query),
        fit_end=str(fit.RENT_APPROVAL_DATE.max().date()), test_start=str(query.RENT_APPROVAL_DATE.min().date()),
        test_end=str(query.RENT_APPROVAL_DATE.max().date()), same_month_test_rows=int(query.RENT_APPROVAL_DATE.eq(fit.RENT_APPROVAL_DATE.max()).sum()),
        selection_basis="Lowest pooled earlier-quarter RMSE among NEW candidates; no primary/public-score tuning",
        formal_promotion_gate_passed=bool(new["eligible"]),
        submission_reason="User explicitly requested one new best-candidate Kaggle trial on 2026-10-19",
        no_test_labels=True, external_resale_or_property_features_used=False, model_reload_verified=True,
        test_order_verified_against_raw=True, output=str(output), output_sha256=sha(output), model_sha256=sha(model_path),
        model_path=str(model_path), prediction_summary={k: float(v) for k, v in saved.Predicted.describe().items()},
        difference_from_previous=dict(mean=float(np.mean(combined-baseline.Predicted)),
            rms=float(np.sqrt(np.mean((combined-baseline.Predicted)**2))), maximum_absolute=float(np.max(abs(combined-baseline.Predicted)))),
        previous_submission_unchanged=sha(base_path) == base_manifest["output_sha256"],
        submitted_to_kaggle=False, expected_pb_before=437.44368)
    dump(report / "submission_manifest.json", audit)
    print(json.dumps({k: v for k, v in audit.items() if k not in ["implementation_sha256", "environment"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
