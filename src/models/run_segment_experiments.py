"""Bounded segment experiments using past expanding-origin OOF residuals.

No competition test inputs, labels or Kaggle submissions are read. Selection
uses 2024 Q3/Q4 only; primary and six-month windows are post-lock diagnostics.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if __package__ in {None, ""}:
    sys.path.insert(0, str(ROOT))

from src.features.segment_calibration import segments, fit_correction, correction_values, available_macro
from src.features.build_stage5_features import aggregate_macro, attach_macro, COE_FEATURES, STOCK_FEATURES
from src.models.run_rmse_improvement import Experiment, Spec, dump, sha, signature, EARLIER
from src.models.run_stage5 import FittedModel, feature_schema, metrics
from src.models.run_tree_tuning import split_inner, search_grid
from src.models.train_baseline import TARGET

BASE = "locked_448_control"
CALIBRATION = {"cal_q1_2024": ("2024-01", "2024-03"), "cal_q2_2024": ("2024-04", "2024-06")}
CALIB_SOURCES = {
    "q3_2024": ["cal_q1_2024", "cal_q2_2024"],
    "q4_2024": ["cal_q2_2024", "q3_2024"],
    "primary_2025": ["q3_2024", "q4_2024"],
    "long_6m": ["cal_q1_2024", "cal_q2_2024"],
}


def fit_light(fit: pd.DataFrame, query: pd.DataFrame, cats: list[str], nums: list[str],
              threads: int, rounds: int | None = None) -> FittedModel:
    from lightgbm import LGBMRegressor, early_stopping
    levels = {c: fit[c].fillna("missing").astype(str).unique().tolist() for c in cats}
    bundle = FittedModel("lightgbm", None, cats, nums, levels)
    params = search_grid()["lightgbm_15_regularized"]["params"]
    model = LGBMRegressor(n_estimators=1000 if rounds is None else rounds, **params,
        objective="regression", max_depth=-1, random_state=42, n_jobs=threads,
        verbosity=-1, deterministic=True, force_col_wise=True)
    kwargs = dict(categorical_feature=cats)
    if rounds is None:
        kwargs.update(eval_X=bundle.matrix(query), eval_y=query[TARGET], eval_metric="rmse",
            callbacks=[early_stopping(80, verbose=False)])
    model.fit(bundle.matrix(fit), fit[TARGET], **kwargs)
    bundle.model = model
    return bundle


class SegmentExperiment:
    def __init__(self, output: Path, threads: int = 3):
        self.output = output.resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        previous = ROOT / "reports/experiments/rmse_improvement_v1"
        old = json.loads((previous / "experiment_manifest.json").read_text())
        self.lock = json.loads((previous / "selection_locked.json").read_text())
        if not old["complete"]:
            raise ValueError("Reference experiment is incomplete")
        for path, digest in old["implementation_sha256"].items():
            if sha(ROOT / path) != digest:
                raise ValueError(f"Reference implementation changed: {path}")
        source = ROOT / old["source"]
        if sha(source) != old["source_sha256"]:
            raise ValueError("Reference training data changed")
        self.engine = Experiment(source, self.output / "base_predictions", threads)
        self.engine.register(BASE, self.lock["weights"])
        self.engine.specs = {name: Spec(**self.lock["specs"][name]) for name in self.lock["weights"]}
        train = self.engine.train
        months = train.RENT_APPROVAL_DATE.dt.to_period("M")
        for name, (start, end) in CALIBRATION.items():
            self.engine.splits[name] = (train.loc[months.lt(pd.Period(start))].copy(),
                train.loc[months.between(pd.Period(start), pd.Period(end))].copy())
        coe_path = ROOT / "data/interim/auxiliary/sg-coe-prices_clean.csv"
        stock_path = ROOT / "data/interim/auxiliary/sg-stock-prices_clean.csv"
        self.monthly, macro = aggregate_macro(pd.read_csv(coe_path), pd.read_csv(stock_path))
        self.grid: dict[str, dict] = {BASE: dict(kind="base")}
        for group, columns in [("global", []), ("flat", ["FLAT_TYPE"]), ("town", ["TOWN"]), ("town_flat", ["TOWN", "FLAT_TYPE"])]:
            for shrink in [100, 500]:
                self.grid[f"bias_{group}_s{shrink}"] = dict(kind="correction", columns=columns, shrink=shrink, trend=False)
        for group, columns in [("global", []), ("flat", ["FLAT_TYPE"]), ("town_flat", ["TOWN", "FLAT_TYPE"])]:
            for shrink in [100, 500]:
                self.grid[f"trend_{group}_s{shrink}"] = dict(kind="correction", columns=columns, shrink=shrink, trend=True)
        implementation = {str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT / "src").rglob("*.py"))}
        self.manifest = dict(source_sha256=old["source_sha256"], implementation_sha256=implementation,
            reference_manifest_sha256=sha(previous / "experiment_manifest.json"),
            reference_lock_sha256=sha(previous / "selection_locked.json"),
            macro_sha256={str(p.relative_to(ROOT)): sha(p) for p in [coe_path, stock_path]},
            environment=self.engine.manifest["environment"], threads=threads, seed=42,
            calibration=CALIB_SOURCES, calibration_window_months=6,
            selection_splits=EARLIER, selection_rule="pooled SSE; incumbent retained unless improvement > 1e-8 RMSE",
            primary_used_for_selection=False, primary_previously_inspected=True,
            diagnostic_splits=["primary_2025", "long_6m"], initial_grid=self.grid.copy(),
            expert=dict(group="FLAT_TYPE", max_groups=4, min_train_rows=3000, min_inner_rows=100,
                selection="largest past OOF SSE, not outer labels", blend_weights=[.25, .5, 1.0], cap=1000),
            macro=dict(aggregation=macro, regimes=["coe_available", "market_available", "market_frozen2"],
                features="housing + COE/stock only; no amenities", blend_weights=[.25, .5, 1.0]),
            test_data_read=False, submission_generated=False,
            limitations=["previously inspected and repeatedly reused validation windows",
                "six-month OOF correction history; forecast-origin/horizon mismatch remains",
                "no new 12-month backtest; old 12-month diagnostic remains applicable only to control",
                "same-month macro is conditional-on-provided-covariates, not a real-time forecast",
                "static HDB snapshot availability", "bounded greedy search, not exhaustive"])
        self.run_id = signature(self.manifest)
        manifest_path = self.output / "experiment_manifest.json"
        if manifest_path.exists() and json.loads(manifest_path.read_text())["run_id"] != self.run_id:
            raise ValueError("Output belongs to a different experiment")
        self.manifest["run_id"] = self.run_id
        self.cache = ROOT / "models/segment_experiments_cache" / self.run_id
        self.cache.mkdir(parents=True, exist_ok=True)
        dump(manifest_path, self.manifest)
        self.predictions, self.rows, self.decisions = {}, [], []

    def base(self, split: str) -> np.ndarray:
        return self.engine.predict(split, BASE)

    def calibration(self, split: str) -> pd.DataFrame:
        parts, provenance = [], []
        fit, query = self.engine.splits[split]
        for name in CALIB_SOURCES[split]:
            history_fit, history_query = self.engine.splits[name]
            if history_fit.RENT_APPROVAL_DATE.max() >= history_query.RENT_APPROVAL_DATE.min():
                raise ValueError("OOF model saw its query dates")
            part = history_query.copy()
            part["prediction"] = self.base(name)
            part["OOF_ORIGIN"] = f"{history_fit.RENT_APPROVAL_DATE.max():%Y-%m}"
            parts.append(part)
            provenance.append(dict(split=name, fit_end=part.OOF_ORIGIN.iloc[0], rows=len(part),
                query_start=f"{part.RENT_APPROVAL_DATE.min():%Y-%m}", query_end=f"{part.RENT_APPROVAL_DATE.max():%Y-%m}"))
        out = pd.concat(parts)
        if out.index.duplicated().any() or out.RENT_APPROVAL_DATE.max() >= query.RENT_APPROVAL_DATE.min():
            raise ValueError("Overlapping or future calibration labels")
        if not out.index.isin(fit.index).all():
            raise ValueError("Calibration must be contained in outer training data")
        dump(self.output / f"calibration_{split}.json", provenance)
        return out

    def expert(self, split: str) -> np.ndarray:
        fit, query = self.engine.splits[split]
        directory = self.cache / f"expert_{split}"
        directory.mkdir(exist_ok=True)
        path = directory / "prediction.npz"
        if path.exists():
            with np.load(path) as saved:
                np.testing.assert_array_equal(saved["source_row"], query.index)
                return saved["prediction"]
        calibration = self.calibration(split)
        errors = (calibration[TARGET] - calibration.prediction) ** 2
        ranking = errors.groupby(calibration.FLAT_TYPE).sum().sort_values(ascending=False)
        groups = [g for g in ranking.index if fit.FLAT_TYPE.eq(g).sum() >= 3000][:4]
        inner_fit, inner = split_inner(fit)
        cats, nums = feature_schema("housing")
        prediction, details = self.base(split).copy(), []
        for group in groups:
            a, b = inner_fit.loc[inner_fit.FLAT_TYPE.eq(group)], inner.loc[inner.FLAT_TYPE.eq(group)]
            selected = fit.loc[fit.FLAT_TYPE.eq(group)]
            mask = query.FLAT_TYPE.eq(group)
            if len(a) < 1000 or len(b) < 100 or not mask.any():
                continue
            chosen = fit_light(a, b, cats, nums, self.engine.threads)
            rounds = max(1, int(chosen.model.best_iteration_))
            bundle = fit_light(selected, query.loc[mask].drop(columns=TARGET), cats, nums, self.engine.threads, rounds)
            prediction[mask.to_numpy()] = bundle.predict(query.loc[mask])
            joblib.dump(bundle, directory / f"{group}.joblib")
            details.append(dict(group=group, rows=len(selected), inner_fit_rows=len(a), inner_rows=len(b), rounds=rounds))
        np.savez_compressed(path, prediction=prediction, source_row=query.index.to_numpy())
        dump(directory / "fit.json", details)
        dump(self.output / f"expert_{split}.json", details)
        return prediction

    def macro(self, split: str, regime: str) -> np.ndarray:
        fit, query = self.engine.splits[split]
        directory = self.cache / f"macro_{split}_{regime}"
        directory.mkdir(exist_ok=True)
        path = directory / "prediction.npz"
        if path.exists():
            with np.load(path) as saved:
                np.testing.assert_array_equal(saved["source_row"], query.index)
                return saved["prediction"]
        cats, nums = feature_schema("housing")
        nums += COE_FEATURES + ([] if regime == "coe_available" else STOCK_FEATURES)
        def transform(a, b):
            if regime == "market_frozen2":
                return attach_macro(a, self.monthly), attach_macro(b, self.monthly, cutoff=a.RENT_APPROVAL_DATE.max())
            return available_macro(a, self.monthly), available_macro(b, self.monthly)
        a, b = transform(*split_inner(fit))
        inner = fit_light(a, b, cats, nums, self.engine.threads)
        rounds = max(1, int(inner.model.best_iteration_))
        a, b = transform(fit, query.drop(columns=TARGET))
        bundle = fit_light(a, b, cats, nums, self.engine.threads, rounds)
        prediction = bundle.predict(b)
        joblib.dump(bundle, directory / "model.joblib")
        np.savez_compressed(path, prediction=prediction, source_row=query.index.to_numpy())
        details = dict(regime=regime, rounds=rounds, train_rows=len(a),
            numeric=nums, query_missing_macro=int(b[COE_FEATURES + ([] if regime == "coe_available" else STOCK_FEATURES)].isna().any(axis=1).sum()),
            query_macro_month_min=b.MACRO_SOURCE_MONTH.min(), query_macro_month_max=b.MACRO_SOURCE_MONTH.max())
        dump(directory / "fit.json", details)
        dump(self.output / f"macro_{split}_{regime}.json", details)
        return prediction

    def predict(self, split: str, candidate: str) -> np.ndarray:
        key = (split, candidate)
        if key in self.predictions:
            return self.predictions[key]
        recipe = self.grid[candidate]
        if recipe["kind"] == "base":
            value = self.base(split)
        elif recipe["kind"] == "correction":
            model = fit_correction(self.calibration(split), recipe["columns"], recipe["shrink"], recipe["trend"])
            value = self.base(split) + correction_values(model, self.engine.splits[split][1].drop(columns=TARGET))
            joblib.dump(model, self.cache / f"correction_{split}_{candidate}.joblib")
        elif recipe["kind"] == "blend":
            incumbent = self.predict(split, recipe["incumbent"])
            other = self.expert(split) if recipe["source"] == "expert" else self.macro(split, recipe["source"])
            value = incumbent * (1 - recipe["weight"]) + other * recipe["weight"]
        else:
            raise ValueError(recipe)
        value = np.maximum(value, 0)
        if not np.isfinite(value).all() or len(value) != len(self.engine.splits[split][1]):
            raise ValueError("Invalid predictions")
        self.predictions[key] = value
        return value

    def evaluate(self, split: str, candidate: str, stage: str):
        if any(r["split"] == split and r["candidate"] == candidate for r in self.rows):
            return
        query = self.engine.splits[split][1]
        row = dict(stage=stage, split=split, candidate=candidate, validation_rows=len(query),
            **metrics(query[TARGET], self.predict(split, candidate)))
        self.rows.append(row)
        pd.DataFrame(self.rows).to_csv(self.output / "results.csv", index=False)
        print(f"SCORE {split} {candidate}: {row['rmse']:.4f}", flush=True)

    def choose(self, stage: str, candidates: list[str], incumbent: str) -> str:
        for split in EARLIER:
            for candidate in candidates:
                self.evaluate(split, candidate, stage)
        results = pd.DataFrame(self.rows)
        results = results.loc[results.split.isin(EARLIER) & results.candidate.isin(candidates)].copy()
        results["sse"] = results.rmse ** 2 * results.validation_rows
        ranked = results.groupby("candidate").agg(sse=("sse", "sum"), rows=("validation_rows", "sum")).reset_index()
        ranked["pooled_rmse"] = np.sqrt(ranked.sse / ranked.rows)
        ranked = ranked.sort_values(["pooled_rmse", "candidate"])
        previous = float(ranked.loc[ranked.candidate.eq(incumbent), "pooled_rmse"].iloc[0])
        best = ranked.iloc[0]
        selected = str(best.candidate) if best.pooled_rmse < previous - 1e-8 else incumbent
        ranked.to_csv(self.output / f"selection_{stage}.csv", index=False)
        self.decisions.append(dict(stage=stage, selected=selected, incumbent=incumbent, candidates=candidates,
            pooled_rmse=float(ranked.loc[ranked.candidate.eq(selected), "pooled_rmse"].iloc[0])))
        dump(self.output / "stage_decisions.json", self.decisions)
        print(f"LOCK {stage}: {selected}", flush=True)
        return selected

    def run(self):
        # Emit priority diagnostics before any new recipe is selected.
        for split in EARLIER:
            fit, query = self.engine.splits[split]
            segments(query, self.base(split), fit).to_csv(self.output / f"segments_control_{split}.csv", index=False)
        incumbent = BASE
        incumbent = self.choose("01_bias", [BASE] + [n for n in self.grid if n.startswith("bias_")], incumbent)
        checkpoints = [BASE, incumbent]
        incumbent = self.choose("02_trend", [incumbent] + [n for n in self.grid if n.startswith("trend_")], incumbent)
        checkpoints.append(incumbent)
        choices = [incumbent]
        for weight in [.25, .5, 1.0]:
            name = f"expert_weight{int(weight * 100)}"
            self.grid[name] = dict(kind="blend", source="expert", weight=weight, incumbent=incumbent)
            choices.append(name)
        incumbent = self.choose("03_experts", choices, incumbent)
        checkpoints.append(incumbent)
        choices = [incumbent]
        for regime in ["coe_available", "market_available", "market_frozen2"]:
            for weight in [.25, .5, 1.0]:
                name = f"{regime}_weight{int(weight * 100)}"
                self.grid[name] = dict(kind="blend", source=regime, weight=weight, incumbent=incumbent)
                choices.append(name)
        incumbent = self.choose("04_macro", choices, incumbent)
        checkpoints.append(incumbent)
        dump(self.output / "selection_locked.json", dict(selected=incumbent, grid=self.grid,
            decisions=self.decisions, checkpoints=checkpoints, primary_used=False))
        for split in ["primary_2025", "long_6m"]:
            for name in dict.fromkeys(checkpoints):
                self.evaluate(split, name, "05_post_lock_diagnostics")
        for split in EARLIER + ["primary_2025", "long_6m"]:
            fit, query = self.engine.splits[split]
            predictions = {n: self.predict(split, n) for n in dict.fromkeys([BASE, incumbent])}
            pd.DataFrame(dict(source_row=query.index, actual=query[TARGET].to_numpy(), **predictions)).to_csv(self.output / f"predictions_{split}.csv", index=False)
            for name, prediction in predictions.items():
                tag = "control" if name == BASE else "selected"
                segments(query, prediction, fit).to_csv(self.output / f"segments_{tag}_{split}.csv", index=False)
        self.manifest.update(complete=True, selected=incumbent, cache=str(self.cache))
        dump(self.output / "experiment_manifest.json", self.manifest)
        self.report(incumbent)
        print(f"COMPLETE {incumbent}", flush=True)

    def report(self, selected: str):
        results = pd.DataFrame(self.rows)
        lines = ["# 分组诊断与针对性优化实验", "", "## 结论与模型选择", "",
            f"锁定方案：`{selected}`。未生成新提交，也未读取竞赛测试集。", "",
            "| 阶段 | 锁定方案 | 较早季度合并 RMSE |", "| --- | --- | ---: |"]
        for row in self.decisions:
            lines.append(f"| {row['stage']} | {row['selected']} | {row['pooled_rmse']:.4f} |")
        lines += ["", "| 窗口 | 方案 | RMSE | MAE | 平均偏差 |", "| --- | --- | ---: | ---: | ---: |"]
        for split in EARLIER + ["primary_2025", "long_6m"]:
            for name in dict.fromkeys([BASE, selected]):
                row = results.loc[results.split.eq(split) & results.candidate.eq(name)].iloc[0]
                lines.append(f"| {split} | {name} | {row.rmse:.4f} | {row.mae:.4f} | {row.bias:+.2f} |")
        lines += ["", "## 优先分组：仅使用较早回测", "",
            "平方误差贡献率分母为该窗口全部样本 SSE。同一分组方案内贡献率和为 100%，不同分组方案互相重叠，不可相加。",
            "偏差取预测减真实值；正值为高估。组内均值偏差的平方贡献仅为描述性分解，不是可实现的降分保证。", "",
            "| 窗口 | 分组 | 样本数 | RMSE | 偏差 | 总 SSE 占比 |", "| --- | --- | ---: | ---: | ---: | ---: |"]
        for split in EARLIER:
            table = pd.read_csv(self.output / f"segments_control_{split}.csv")
            for row in table.loc[table.partition.eq("town_flat") & table.rows.ge(100)].head(6).itertuples():
                lines.append(f"| {split} | {row.segment} | {row.rows} | {row.rmse:.2f} | {row.bias:+.2f} | {row.sse_share:.1%} |")
        lines += ["", "## 方法与防泄漏约定", "",
            "- 按城镇、房型、区域、面积、房龄、月份、历史地址样本数及交叉组统计 RMSE、MAE、偏差、P90 绝对误差、SSE 贡献率和大误差数量。",
            "- 修正器只使用预测窗口之前六个月的扩展时间 OOF 残差。每条校准预测的基础模型仅使用该季度之前的标签；不使用训练内拟合误差。",
            "- 常数修正向零收缩；趋势修正拟合组内月度 OOF 残差，斜率限制为每月 ±25，新窗口增长以六个月时间常数衰减，最终修正限制为 ±200。",
            "- 小样本组不修正，未见组回退原全局模型。分段模型按历史 OOF 的 SSE 选最多四种房型；每组至少 3000 条外层训练样本，内部验证至少 100 条，迭代轮数只用内部时间切分选择。",
            "- 分段及宏观 LightGBM 均使用同一 15 叶正则化配置，与当前方案以 25%/50%/100% 权重比较。",
            "- 对比对应月份已提供的 COE、COE+股票，以及滞后两个月并冻结截止时点的 COE+股票。同期辅助信息条件下的回测不是严格实时预测；不涉及验证租金标签。",
            "- 各阶段按 2024 年 Q3/Q4 合并 SSE 贪心选择，始终保留上一方案。锁定后才计算 2025 年 Q1 和六个月窗口；所有这些时期先前已被查看，不能称为全新独立测试。",
            "- 本轮不重跑十二个月窗口；不能将旧方案的十二个月结果移用于本轮新方案，也不能据本地改进推断 Kaggle 一定改善。",
            "- 原始数据、原实验目录、已提交文件均不覆盖；逐行标签/预测及模型只保存在本地。", ""]
        (self.output / "segment_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/experiments/segment_v1")
    parser.add_argument("--threads", type=int, default=3)
    args = parser.parse_args()
    SegmentExperiment(args.output_dir, args.threads).run()
