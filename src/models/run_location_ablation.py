"""Compare Ridge feature groups on fixed primary, earlier temporal and random splits.

No test labels, leaderboard feedback or validation-fitted transformations enter
training. All experiments use the original Ridge settings. This command does
not overwrite baseline_results.csv or generate a competition submission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if __package__ in {None, ""}:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.train_baseline import (
    CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET, build_ridge_pipeline, rmse, split_metadata,
)

HDB_CATEGORICAL = ["SUBZONE", "PLANNING_AREA", "REGION"]
HDB_NUMERIC = [
    "LATITUDE", "LONGITUDE", "YEAR_COMPLETED", "MAX_FLOOR", "BUILDING_AGE",
    "YEAR_COMPLETED_MISSING", "BUILDING_AGE_MISSING", "ADDRESS_MATCHED", "LOCATION_COORDINATES_VALID",
]
MRT_NUMERIC = ["NEAREST_OPEN_MRT_DISTANCE_KM", "OPEN_MRT_COUNT_WITHIN_500M", "OPEN_MRT_COUNT_WITHIN_1000M"]
SCHOOL_NUMERIC = ["NEAREST_SCHOOL_DISTANCE_KM", "SCHOOL_COUNT_WITHIN_500M", "SCHOOL_COUNT_WITHIN_1000M"]
MALL_NUMERIC = ["NEAREST_MALL_DISTANCE_KM", "MALL_COUNT_WITHIN_500M", "MALL_COUNT_WITHIN_1000M"]
VARIANTS = {
    "main_only": ([], []),
    "hdb": (HDB_CATEGORICAL, HDB_NUMERIC),
    "hdb_mrt": (HDB_CATEGORICAL, HDB_NUMERIC + MRT_NUMERIC),
    "hdb_mrt_school": (HDB_CATEGORICAL, HDB_NUMERIC + MRT_NUMERIC + SCHOOL_NUMERIC),
    "hdb_mrt_school_mall": (HDB_CATEGORICAL, HDB_NUMERIC + MRT_NUMERIC + SCHOOL_NUMERIC + MALL_NUMERIC),
    "hdb_school_mall": (HDB_CATEGORICAL, HDB_NUMERIC + SCHOOL_NUMERIC + MALL_NUMERIC),
}
LABELS = {
    "main_only": "Main-table Ridge",
    "hdb": "+ HDB coordinates and block attributes",
    "hdb_mrt": "+ HDB + MRT",
    "hdb_mrt_school": "+ HDB + MRT + schools",
    "hdb_mrt_school_mall": "+ HDB + MRT + schools + malls",
    "hdb_school_mall": "+ HDB + schools + malls (no MRT)",
}


def feature_schema(variant: str) -> tuple[list[str], list[str]]:
    extra_categories, extra_numeric = VARIANTS[variant]
    return CATEGORICAL_FEATURES + extra_categories, NUMERIC_FEATURES + extra_numeric


def load_training(path: Path, variants: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    categorical, numeric = set(), set()
    for variant in variants:
        cats, nums = feature_schema(variant)
        categorical.update(cats)
        numeric.update(nums)
    required = categorical | numeric | {TARGET, "RENT_APPROVAL_DATE"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing input features: {', '.join(missing)}")
    for column in categorical:
        df[column] = df[column].fillna("missing").astype(str)
    for column in numeric:
        df[column] = pd.to_numeric(df[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    if not np.isfinite(df[TARGET]).all():
        raise ValueError("Missing or non-finite training rent labels")
    df["RENT_APPROVAL_DATE"] = pd.to_datetime(df["RENT_APPROVAL_DATE"], format="ISO8601", errors="coerce")
    if df["RENT_APPROVAL_DATE"].isna().any():
        raise ValueError("Missing or invalid approval dates")
    return df


def validation_splits(
    df: pd.DataFrame, validation_months: int, backtest_windows: int, include_random: bool,
) -> list[tuple[str, pd.DataFrame, pd.DataFrame]]:
    if validation_months < 1 or backtest_windows < 0:
        raise ValueError("validation_months must be positive and backtest_windows non-negative")
    periods = df["RENT_APPROVAL_DATE"].dt.to_period("M")
    months = sorted(periods.unique())
    if len(months) <= validation_months * (backtest_windows + 1):
        raise ValueError("Not enough approval months for the requested temporal backtests")
    splits = []
    for offset in range(backtest_windows + 1):
        stop = len(months) - offset * validation_months
        selected = months[stop - validation_months:stop]
        fit = df.loc[periods < selected[0]].copy()
        valid = df.loc[periods.isin(selected)].copy()
        name = "temporal" if offset == 0 else f"temporal_backtest_{offset}"
        splits.append((name, fit, valid))
    if include_random:
        fit, valid = train_test_split(df, test_size=0.2, random_state=42)
        splits.append(("random", fit, valid))
    return splits


def add_comparisons(results: pd.DataFrame) -> pd.DataFrame:
    out = results.copy()
    baseline = out.loc[out["variant"].eq("main_only")].set_index("split")["rmse"]
    out["baseline_rmse"] = out["split"].map(baseline)
    out["rmse_improvement"] = out["baseline_rmse"] - out["rmse"]
    out["improvement_pct"] = 100 * out["rmse_improvement"] / out["baseline_rmse"]
    return out


def group_errors(train: pd.DataFrame, predictions: dict[str, pd.Series], variants: list[str]) -> pd.DataFrame:
    groups = [c for c in ["TOWN", "FLAT_TYPE", "RENT_APPROVAL_DATE", "REGION"] if c in train]
    rows = []
    for variant in dict.fromkeys(variants):
        series = predictions[variant]
        frame = train.loc[series.index, groups + [TARGET]].copy()
        frame["_error"] = series.to_numpy() - frame[TARGET].to_numpy()
        for column in groups:
            for value, group in frame.groupby(column, dropna=False, sort=True):
                error = group["_error"].to_numpy()
                rows.append({
                    "variant": variant, "group_field": column, "group_value": str(value),
                    "rows": len(group), "rmse": float(np.sqrt(np.mean(error ** 2))),
                    "mae": float(np.mean(np.abs(error))), "bias_predicted_minus_actual": float(np.mean(error)),
                })
    return pd.DataFrame(rows)


def render_report(results: pd.DataFrame, manifest: dict) -> str:
    primary = results.loc[results["split"].eq("temporal")]
    baseline = primary.loc[primary["variant"].eq("main_only"), "rmse"].iloc[0]
    lines = ["# Location Feature Ridge Ablation", "", "## Design", ""]
    for split in manifest["splits"]:
        lines.append(f"- {split['split']}: train {split['fit_start']} to {split['fit_end']} ({split['fit_rows']:,} rows), validate {split['validation_start']} to {split['validation_end']} ({split['validation_rows']:,} rows).")
    lines.extend([
        "", "All variants use the same Ridge parameters and fixed splits; imputer, scaler and one-hot encoder are fitted on each training fold only.", "",
        "## Main temporal validation", "",
        "| Feature set | Fields | RMSE | Improvement over this run's main-table baseline | Improvement % |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for _, row in primary.iterrows():
        lines.append(f"| {LABELS[row['variant']]} | {row['feature_count']} | {row['rmse']:.4f} | {row['rmse_improvement']:+.4f} | {row['improvement_pct']:+.2f}% |")
    lines.extend(["", "## Earlier windows and random split", "", "| Feature set | Split | RMSE | Improvement over same-split baseline |", "| --- | --- | ---: | ---: |"])
    for _, row in results.loc[~results["split"].eq("temporal")].iterrows():
        lines.append(f"| {LABELS[row['variant']]} | {row['split']} | {row['rmse']:.4f} | {row['rmse_improvement']:+.4f} |")
    lines.extend(["", "## Interpretation and limitations", ""])
    reference = manifest.get("previous_reference_rmse")
    if reference is not None:
        lines.append(f"- The previously recorded Ridge temporal RMSE is {reference:.4f}; this rerun with the same configuration gives {baseline:.4f}. Environments differ, so all improvements use this same-environment rerun as reference.")
    best = manifest["best_primary_variant"]
    without_mrt = manifest["best_primary_variant_without_mrt"]
    best_row = primary.loc[primary["variant"].eq(best)].iloc[0]
    baseline_row = primary.loc[primary["variant"].eq("main_only")].iloc[0]
    earlier = results.loc[results["split"].str.startswith("temporal_backtest_") & results["variant"].eq(best)]
    lines.extend([
        f"- Lowest main temporal validation score: {best}; lowest without MRT: {without_mrt}.",
        f"- The best main-validation variant lowers RMSE by {best_row['rmse_improvement']:.4f} ({best_row['improvement_pct']:.2f}%) against this baseline, a limited gain. It beats the matching baseline in {int(earlier['rmse_improvement'].gt(0).sum())} of {len(earlier)} earlier windows.",
        f"- Mean prediction minus actual rent on the main validation: baseline {baseline_row['bias_predicted_minus_actual']:+.2f}, lowest-RMSE variant {best_row['bias_predicted_minus_actual']:+.2f}. Positive values mean overall overprediction, which location features do not fix.",
        "- The lowest score only selects candidates for later experiments; it does not replace independent test performance or show statistical significance. Earlier windows check stability; the random split is auxiliary.",
        f"- MRT temporal mode: {manifest['mrt_mode']}. With snapshot or unknown, MRT results are exploratory because currently open stations may not have been open in early rental months.",
        "- School, mall and some block attributes are also static snapshots, so removing MRT does not prove the other auxiliary fields are historically available.",
        "- Validation windows are 3 months by default while the test set extends further, so long-horizon extrapolation error may be larger.",
        "- ADDRESS_KEY and POSTAL_CODE are not used as model inputs. Constant flags such as address-match indicators stay in the preset variants; Ridge regularisation and scaling match the baseline.",
        "- No hyperparameters were tuned, baseline_config.json was not modified, and no Kaggle submission was generated.", "",
        "Group errors are in group_errors.csv: RMSE, MAE and signed bias by town, flat type, month and region for the baseline and main candidates.", "",
        "## Next experiments", "",
        "First compare models or time features that capture nonlinear time changes, and check whether overall overprediction relates to long-term linear trend extrapolation. This is only a hypothesis and cannot be inferred from mean bias alone. Keep the same windows and keep MRT as a separate control with its availability caveat.", "",
    ])
    return "\n".join(lines)


def run_experiments(
    train_path: Path, output_dir: Path, location_manifest: Path,
    variants: list[str], validation_months: int = 3, backtest_windows: int = 2,
    include_random: bool = True, reference_rmse: float | None = None,
) -> tuple[pd.DataFrame, dict]:
    if "main_only" not in variants or len(set(variants)) != len(variants) or not set(variants).issubset(VARIANTS):
        raise ValueError("Select distinct known variants and include main_only for a fair baseline")
    train_path, output_dir = train_path.resolve(), output_dir.resolve()
    if output_dir == train_path.parent or output_dir.is_relative_to(PROJECT_ROOT / "data/raw"):
        raise ValueError("Keep experiment outputs outside source data directories")
    train = load_training(train_path, variants)
    splits = validation_splits(train, validation_months, backtest_windows, include_random)
    provenance = json.loads(location_manifest.read_text(encoding="utf-8")) if location_manifest.is_file() else {}
    manifest = {
        "model": "sklearn_ridge_fallback", "configuration": {
            "alpha": 10.0, "solver": "lsqr", "tol": 1e-4, "fit_intercept": False,
            "one_hot_min_frequency": 2, "numeric_imputer": "median", "numeric_scaler": "StandardScaler",
        },
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__},
        "train_sha256": hashlib.sha256(train_path.read_bytes()).hexdigest(),
        "location_manifest_sha256": hashlib.sha256(location_manifest.read_bytes()).hexdigest() if location_manifest.is_file() else None,
        "mrt_mode": provenance.get("mrt_mode", "unknown"),
        "previous_reference_rmse": reference_rmse,
        "validation_months": validation_months, "random_seed": 42, "random_test_size": 0.2,
        "splits": [], "feature_sets": {},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    records, primary_predictions = [], {}
    for split_name, fit, valid in splits:
        metadata = split_metadata(split_name, fit, valid)
        manifest["splits"].append(metadata)
        for variant in variants:
            cats, nums = feature_schema(variant)
            features = cats + nums
            manifest["feature_sets"][variant] = {"categorical": cats, "numeric": nums, "uses_mrt": any(c in MRT_NUMERIC for c in nums)}
            print(f"Training {split_name}: {variant} ({len(features)} features)", flush=True)
            start = time.perf_counter()
            model = build_ridge_pipeline(cats, nums)
            model.fit(fit[features], fit[TARGET])
            prediction = np.maximum(model.predict(valid[features]), 0)
            score = rmse(valid[TARGET], prediction)
            records.append({
                **metadata, "variant": variant, "model": "sklearn_ridge_fallback", "feature_count": len(features),
                "rmse": score, "mae": float(mean_absolute_error(valid[TARGET], prediction)),
                "bias_predicted_minus_actual": float(np.mean(prediction - valid[TARGET].to_numpy())),
                "seconds": time.perf_counter() - start,
            })
            if split_name == "temporal":
                primary_predictions[variant] = pd.Series(prediction, index=valid.index)
            print(f"  RMSE = {score:.4f}", flush=True)
            add_comparisons(pd.DataFrame(records)).to_csv(output_dir / "ablation_results.csv", index=False)
    results = add_comparisons(pd.DataFrame(records))
    primary = results.loc[results["split"].eq("temporal")]
    best = primary.loc[primary["rmse"].idxmin(), "variant"]
    no_mrt = primary.loc[~primary["variant"].map(lambda v: manifest["feature_sets"][v]["uses_mrt"])]
    best_no_mrt = no_mrt.loc[no_mrt["rmse"].idxmin(), "variant"]
    manifest["best_primary_variant"] = best
    manifest["best_primary_variant_without_mrt"] = best_no_mrt
    error_table = group_errors(train, primary_predictions, ["main_only", best, best_no_mrt])
    error_table.to_csv(output_dir / "group_errors.csv", index=False)
    # Per-record validation predictions remain a private CSV ignored by Git.
    pd.DataFrame({"source_row": splits[0][2].index, "actual": splits[0][2][TARGET], **primary_predictions}).to_csv(output_dir / "primary_validation_predictions.csv", index=False)
    (output_dir / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    (output_dir / "location_ablation_report.md").write_text(render_report(results, manifest), encoding="utf-8")
    return results, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-path", type=Path, default=PROJECT_ROOT / "data/processed/train_location_features.csv")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "reports/experiments/location_ablation_v1")
    parser.add_argument("--location-manifest", type=Path, default=PROJECT_ROOT / "reports/location_features/location_features_manifest.json")
    parser.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS))
    parser.add_argument("--validation-months", type=int, default=3)
    parser.add_argument("--backtest-windows", type=int, default=2)
    parser.add_argument("--skip-random", action="store_true")
    parser.add_argument("--reference-rmse", type=float, help="Optional previous primary RMSE, for context only")
    args = parser.parse_args()
    try:
        results, _ = run_experiments(args.train_path, args.output_dir, args.location_manifest, args.variants, args.validation_months, args.backtest_windows, not args.skip_random, args.reference_rmse)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    print(results[["split", "variant", "rmse", "rmse_improvement"]].to_string(index=False))
    print(f"Report: {args.output_dir / 'location_ablation_report.md'}")


if __name__ == "__main__":
    main()
