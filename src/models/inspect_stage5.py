"""Independently check exported stage-5 scores and make report figures."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if __package__ in {None, ""}:
    sys.path.insert(0, str(ROOT))

from src.models.run_stage5 import FittedModel, cache_identity, implementation_hashes, load_training, render_report, select_candidate
from src.features.build_stage5_features import aggregate_macro, attach_macro


def make_figures(output: Path, results: pd.DataFrame, manifest: dict, importance: pd.DataFrame) -> list[str]:
    (ROOT / ".matplotlib").mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    directory = output / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    selected = manifest["selected_candidate"]
    primary = results.loc[results.split.eq("temporal")].sort_values("rmse", ascending=False)
    colors = ["#267F75" if row.candidate == selected else "#C8D3DD" if row.scheme != "mrt_snapshot" else "#E6A05A" for row in primary.itertuples()]
    fig, ax = plt.subplots(figsize=(12, 9))
    bars = ax.barh(primary.candidate, primary.rmse, color=colors)
    ax.bar_label(bars, fmt="%.2f", padding=4, fontsize=9)
    ax.set_xlabel("Validation RMSE (SGD per month)")
    ax.set_title("Stage 5: January–March 2025 validation\nGreen: selected on earlier backtests; orange: MRT snapshot experiments")
    ax.set_xlim(0, primary.rmse.max() * 1.15)
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(directory / "01_model_comparison.png", dpi=160)
    plt.close(fig)

    importance = importance.sort_values("rmse_increase_mean")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(importance.group, importance.rmse_increase_mean, color="#267F75")
    ax.set_xlabel("Increase in sample RMSE after grouped permutation")
    ax.set_title(f"Selected candidate: {selected}\nFixed validation sample, two permutations per group")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(directory / "02_group_importance.png", dpi=160)
    plt.close(fig)

    groups = pd.read_csv(output / "group_errors.csv")
    months = groups.loc[groups.field.eq("RENT_APPROVAL_DATE")]
    fig, ax = plt.subplots(figsize=(9, 5))
    for candidate in dict.fromkeys(["ridge_main", "ridge_spline_main", selected]):
        subset = months.loc[months.candidate.eq(candidate)].sort_values("value")
        ax.plot(subset.value.str[:7], subset.bias, marker="o", label=candidate)
    ax.axhline(0, color="#666666", linewidth=1)
    ax.set_ylabel("Mean prediction − actual rent (SGD/month)")
    ax.set_xlabel("Approval month")
    ax.set_title("Monthly prediction bias on the primary validation window")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(directory / "03_monthly_bias.png", dpi=160)
    plt.close(fig)
    return [f"figures/{filename}" for filename in ["01_model_comparison.png", "02_group_importance.png", "03_monthly_bias.png"]]


def inspect_outputs(output: Path, render_figures: bool = True) -> dict:
    results = pd.read_csv(output / "results.csv")
    manifest = json.loads((output / "experiment_manifest.json").read_text(encoding="utf-8"))
    train = pd.read_csv(ROOT / "data/processed/train_location_features.csv")
    if results.duplicated(["split", "candidate"]).any():
        raise ValueError("Duplicate experiment rows")
    assert np.isfinite(results[["rmse", "mae", "bias"]]).all().all()
    for relative, digest in manifest["source_sha256"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest
    selection, table = select_candidate(results)
    assert selection == manifest["selected_candidate"]
    lock = json.loads((output / "selection_locked.json").read_text(encoding="utf-8"))
    assert lock["selected_candidate"] == selection and lock["primary_used"] is False
    checked_predictions = 0
    for split in ["temporal_backtest_2", "temporal_backtest_1", "temporal"]:
        frame = pd.read_csv(output / f"predictions_{split}.csv")
        actual = train.loc[frame.source_row, "MONTHLY_RENT"].to_numpy()
        np.testing.assert_array_equal(actual, frame.actual)
        subset = results.loc[results.split.eq(split)]
        assert len(frame) == subset.validation_rows.iloc[0]
        for row in subset.itertuples():
            assert row.fit_end < row.validation_start
            if row.family in {"catboost", "lightgbm"}:
                assert row.inner_fit_end < row.inner_validation_start
                assert row.inner_validation_end <= row.fit_end
                assert 0 < row.iterations <= manifest["tree_iterations_cap"]
            error = frame[row.candidate] - actual
            assert np.isclose(row.rmse, np.sqrt(np.mean(error ** 2)), rtol=1e-12)
            assert np.isclose(row.mae, np.mean(np.abs(error)), rtol=1e-12)
            assert np.isclose(row.bias, np.mean(error), rtol=1e-12)
            if row.family == "blend":
                expected = (frame[f"catboost_{row.scheme}"] + frame[f"lightgbm_{row.scheme}"]) / 2
                np.testing.assert_allclose(frame[row.candidate], expected, rtol=1e-12)
        checked_predictions += len(frame) * len(subset)
    groups = pd.read_csv(output / "group_errors.csv")
    primary = results.loc[results.split.eq("temporal")].set_index("candidate")
    for (candidate, field), frame in groups.groupby(["candidate", "field"]):
        assert frame.rows.sum() == primary.loc[candidate, "validation_rows"]
        assert np.isclose(np.sqrt(np.average(frame.rmse ** 2, weights=frame.rows)), primary.loc[candidate, "rmse"], rtol=1e-12)
    importance = pd.read_csv(output / "group_permutation_importance.csv")
    # Verify that the saved diagnostic models reproduce their exported values.
    monthly, _ = aggregate_macro(
        pd.read_csv(ROOT / "data/interim/auxiliary/sg-coe-prices_clean.csv"),
        pd.read_csv(ROOT / "data/interim/auxiliary/sg-stock-prices_clean.csv"),
    )
    exported = pd.read_csv(output / "predictions_temporal.csv")
    prepared = load_training(ROOT / "data/processed/train_location_features.csv")
    query = attach_macro(prepared.loc[exported.source_row], monthly, pd.Timestamp(manifest["diagnostic_model_training_end"]))
    for component in manifest["selected_components"]:
        directory = ROOT / "models/stage5"
        schema = json.loads((directory / f"{component}.schema.json").read_text(encoding="utf-8"))
        if schema["family"] == "catboost":
            from catboost import CatBoostRegressor
            model = CatBoostRegressor()
            model.load_model(str(directory / f"{component}.cbm"))
        elif schema["family"] == "lightgbm":
            from lightgbm import Booster
            model = Booster(model_file=str(directory / f"{component}.txt"))
        else:
            import joblib
            model = joblib.load(directory / f"{component}.joblib")
        bundle = FittedModel(schema["family"], model, schema["cats"], schema["nums"], schema["levels"])
        np.testing.assert_allclose(bundle.predict(query), exported[component], rtol=1e-10, atol=1e-7)
    figures = make_figures(output, results, manifest, importance) if render_figures else [f"figures/{name}" for name in ["01_model_comparison.png", "02_group_importance.png", "03_monthly_bias.png"]]
    assert all((output / figure).is_file() for figure in figures)
    if "implementation_sha256" not in manifest:
        # The first run predates automatic implementation fingerprinting. It
        # has just passed score/model reload checks. Preserve its completed
        # caches under the finalized implementation identity without retraining.
        old_identity = manifest["cache_signature"]
        manifest["implementation_sha256"] = implementation_hashes()
        new_identity = cache_identity(manifest)
        cache_root = (ROOT / "models/stage5_cache").resolve()
        old_directory = (cache_root / old_identity).resolve()
        new_directory = (cache_root / new_identity).resolve()
        assert old_directory.parent == cache_root and new_directory.parent == cache_root
        if old_directory != new_directory and old_directory.is_dir() and not new_directory.exists():
            old_directory.rename(new_directory)
        manifest["cache_signature"] = new_identity
        (output / "experiment_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    assert cache_identity(manifest) == manifest["cache_signature"]
    assert (ROOT / "models/stage5_cache" / manifest["cache_signature"]).is_dir()
    text = render_report(results, manifest, importance)
    selected_groups = groups.loc[groups.candidate.eq(selection)]
    text += "\n## 候选模型的分组误差诊断\n\n下表仅为主验证描述，不据此重新挑选模型。城镇按至少 100 条记录筛选后列出误差最高的 4 组，避免按极小样本排序。\n\n"
    text += "| 城镇 | 验证记录数 | RMSE | 平均预测偏差 |\n| --- | ---: | ---: | ---: |\n"
    towns = selected_groups.loc[selected_groups.field.eq("TOWN") & selected_groups.rows.ge(100)].nlargest(4, "rmse")
    for row in towns.itertuples():
        text += f"| {row.value} | {row.rows} | {row.rmse:.2f} | {row.bias:+.2f} |\n"
    text += "\n| 房型 | 验证记录数 | RMSE | 平均预测偏差 |\n| --- | ---: | ---: | ---: |\n"
    for row in selected_groups.loc[selected_groups.field.eq("FLAT_TYPE")].itertuples():
        text += f"| {row.value} | {row.rows} | {row.rmse:.2f} | {row.bias:+.2f} |\n"
    text += "\n后续应检查高误差城镇与大户型的残差分布，并在较早时间窗口验证任何修正方案。稀有房型样本少，不宜仅按主验证分数判断模型优劣。\n"
    native = pd.read_csv(output / "native_feature_importance.csv")
    text += "\n## 选定组件的训练重要性\n\n| 组件 | 特征 | 重要性数值 | 方法 |\n| --- | --- | ---: | --- |\n"
    for component in manifest["selected_components"]:
        for row in native.loc[native.candidate.eq(component)].nlargest(5, "importance").itertuples():
            text += f"| {component} | {row.feature} | {row.importance:.4g} | {row.method} |\n"
    text += "\n训练期的时间字段重要性较高，而主验证局部置换时间组的作用较小，两者衡量范围不同，不能将局部重要性解释为时间趋势不重要。\n"
    text += "\n## 结果图片\n\n" + "\n\n".join(f"![{Path(figure).stem}]({figure})" for figure in figures) + "\n"
    (output / "stage5_report.md").write_text(text, encoding="utf-8")
    check = {"experiment_rows": len(results), "verified_prediction_values": checked_predictions, "source_hashes_match": True, "selected_candidate": selection, "selection_uses_only_earlier_windows": True, "diagnostic_model_reload_matches": True, "figures": figures}
    (output / "output_validation.json").write_text(json.dumps(check, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(check, indent=2))
    return check


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/experiments/stage5_v1")
    parser.add_argument("--skip-figures", action="store_true", help="Keep already-inspected figures when only refreshing report text")
    args = parser.parse_args()
    inspect_outputs(args.output_dir, not args.skip_figures)
