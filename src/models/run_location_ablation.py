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
    "main_only": "原主表 Ridge",
    "hdb": "+ HDB 坐标与楼栋属性",
    "hdb_mrt": "+ HDB + MRT",
    "hdb_mrt_school": "+ HDB + MRT + 学校",
    "hdb_mrt_school_mall": "+ HDB + MRT + 学校 + 商场",
    "hdb_school_mall": "+ HDB + 学校 + 商场（无 MRT）",
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
    lines = ["# 位置特征 Ridge 消融实验", "", "## 实验设计", ""]
    for split in manifest["splits"]:
        lines.append(f"- {split['split']}: {split['fit_start']} 至 {split['fit_end']} 训练（{split['fit_rows']:,} 行），{split['validation_start']} 至 {split['validation_end']} 验证（{split['validation_rows']:,} 行）。")
    lines.extend([
        "", "所有方案使用相同 Ridge 参数和固定划分，填补器、标准化和 One-hot 编码仅在各训练折拟合。", "",
        "## 主要时间验证结果", "",
        "| 特征方案 | 字段数 | RMSE | 相对本次主表基线改善 | 改善比例 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for _, row in primary.iterrows():
        lines.append(f"| {LABELS[row['variant']]} | {row['feature_count']} | {row['rmse']:.4f} | {row['rmse_improvement']:+.4f} | {row['improvement_pct']:+.2f}% |")
    lines.extend(["", "## 更早时间窗口与随机切分", "", "| 特征方案 | 验证划分 | RMSE | 相对同划分主表基线改善 |", "| --- | --- | ---: | ---: |"])
    for _, row in results.loc[~results["split"].eq("temporal")].iterrows():
        lines.append(f"| {LABELS[row['variant']]} | {row['split']} | {row['rmse']:.4f} | {row['rmse_improvement']:+.4f} |")
    lines.extend(["", "## 结果解释与限制", ""])
    reference = manifest.get("previous_reference_rmse")
    if reference is not None:
        lines.append(f"- 先前记录的 Ridge 时间 RMSE 为 {reference:.4f}；本次相同配置复跑为 {baseline:.4f}。运行环境不同，所有改善均以本次同环境复跑为参照。")
    best = manifest["best_primary_variant"]
    without_mrt = manifest["best_primary_variant_without_mrt"]
    best_row = primary.loc[primary["variant"].eq(best)].iloc[0]
    baseline_row = primary.loc[primary["variant"].eq("main_only")].iloc[0]
    earlier = results.loc[results["split"].str.startswith("temporal_backtest_") & results["variant"].eq(best)]
    lines.extend([
        f"- 本次主要时间验证得分最低的方案为 {best}；未使用 MRT 的方案中得分最低的是 {without_mrt}。",
        f"- 主验证最佳方案相对本次基线降低 RMSE {best_row['rmse_improvement']:.4f}（{best_row['improvement_pct']:.2f}%），提升有限。它在 {len(earlier)} 个更早窗口中有 {int(earlier['rmse_improvement'].gt(0).sum())} 个窗口优于对应基线。",
        f"- 主验证集平均预测减真实租金：主表基线 {baseline_row['bias_predicted_minus_actual']:+.2f}，最低 RMSE 方案 {best_row['bias_predicted_minus_actual']:+.2f}。正值表示整体高估，位置特征尚未解决这一偏差。",
        "- 上述最低分只用于确定后续实验候选，不能替代独立测试集表现或证明统计显著性。更早时间窗口用于检查表现稳定性，随机切分仅为辅助。",
        f"- MRT 时间策略：{manifest['mrt_mode']}。若为 snapshot 或 unknown，含 MRT 的结果仅为探索，当前 open 站点可能在早期租赁月份尚未开放。",
        "- 学校、商场及部分楼栋属性也来自静态快照，移除 MRT 不能证明其他辅助字段完全满足历史可用性。",
        "- 验证窗口为 3 个月（默认），测试集延伸至更远月份，长期外推误差可能更大。",
        "- 不使用 ADDRESS_KEY 或 POSTAL_CODE 做新模型输入。地址匹配标记等常量保留在预设方案内，Ridge 正则化和标准化处理与基线相同。",
        "- 未调节模型参数，未修改 baseline_config.json，未生成或上传 Kaggle 提交。", "",
        "分组误差见 group_errors.csv，包含主表基线与主要候选按城镇、房型、月份及区域的 RMSE、MAE 和有方向偏差。", "",
        "## 下一步实验", "",
        "优先比较能够表达非线性时间变化的模型或时间特征方案，检查预测整体高估是否与长期线性趋势外推有关。这个原因目前仅为待验证假设，不能从平均偏差直接推断。继续使用同样的时间窗口，MRT 保留为带历史可用性限制的独立对照。", "",
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
