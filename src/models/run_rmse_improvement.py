"""Sequential, resumable temporal experiments; never uses test labels/data.

1. CatBoost caps 800/1600/2400, fixed LightGBM control.
2. LightGBM screens windows 24/36 and half-lives 12/24; confirm its best new
   recipe with CatBoost and retain the incumbent unless the blend improves.
3. LightGBM screens causal rent histories of 6/12 months; confirm the best new
   recipe with CatBoost, again retaining the incumbent as an eligible option.
4. Static LightGBM residuals + flat/damped monthly trend; fixed 25/50/75% blends.

Each decision uses Jul-Sep and Oct-Dec 2024 pooled squared error. Decisions are
locked before Jan-Mar 2025 and 6/12-month diagnostic backtests are evaluated.
These quarters have already been inspected; they are not pristine holdouts.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if __package__ in {None, ""}:
    sys.path.insert(0, str(ROOT))

from src.features.build_temporal_features import (
    fit_trend, historical_features, history_columns, month_number, trend_values,
)
from src.models.run_stage5 import FittedModel, feature_schema, load_training, metrics, grouped_errors
from src.models.run_tree_tuning import search_grid, split_inner
from src.models.train_baseline import TARGET, split_metadata

EARLIER = ["q3_2024", "q4_2024"]
SPLITS = {
    "q3_2024": ("2024-07", "2024-09"),
    "q4_2024": ("2024-10", "2024-12"),
    "primary_2025": ("2025-01", "2025-03"),
    "long_6m": ("2024-07", "2024-12"),
    "long_12m": ("2024-01", "2024-12"),
}


def dump(path: Path, value: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def signature(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:20]


@dataclass(frozen=True)
class Spec:
    family: str
    cap: int = 800
    window: int = 0
    half_life: int = 0
    history: int = 0
    trend: str = ""

    @property
    def name(self) -> str:
        return f"{self.family}_cap{self.cap}_w{self.window}_h{self.half_life}_hist{self.history}_{self.trend or 'direct'}"


def select_training(fit: pd.DataFrame, spec: Spec) -> pd.DataFrame:
    if spec.window:
        months = month_number(fit)
        return fit.loc[months > months.max() - spec.window].copy()
    return fit.copy()


def sample_weights(fit: pd.DataFrame, spec: Spec) -> np.ndarray | None:
    if not spec.half_life:
        return None
    months = month_number(fit)
    weights = np.power(0.5, (months.max() - months).to_numpy() / spec.half_life)
    return weights / weights.mean()


def prepare(fit: pd.DataFrame, query: pd.DataFrame, spec: Spec):
    # Remove query labels before any transform, early stopping excepted at fit().
    query = query.drop(columns=TARGET, errors="ignore").copy()
    fit = fit.copy()
    cats, nums = feature_schema("housing")
    if spec.history:
        fit, query = historical_features(fit, query, spec.history)
        nums += history_columns(spec.history)
    trend = None
    if spec.trend:
        trend = fit_trend(fit, spec.trend)
        fit[TARGET] = fit[TARGET] - trend_values(trend, fit)
        # Calendar level and deterministic ageing are handled by the trend.
        nums = [c for c in nums if c not in {
            "RENT_YEAR", "RENT_MONTH", "RENT_MONTH_INDEX", "FLAT_AGE", "BUILDING_AGE",
            "RENT_MONTH_SIN", "RENT_MONTH_COS",
        }]
        cats = [c for c in cats if c != "FLAT_AGE_BAND"]
    return fit, query, cats, nums, trend


def score_choice(rows: list[dict], names: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame = frame.loc[frame.split.isin(EARLIER) & frame.candidate.isin(names)].copy()
    if frame.duplicated(["split", "candidate"]).any():
        raise ValueError("Duplicate selection results")
    if set(frame.candidate) != set(names) or not frame.groupby("candidate").split.agg(set).map(lambda s: s == set(EARLIER)).all():
        raise ValueError("Every choice needs both earlier quarters")
    frame["sse"] = frame.rmse ** 2 * frame.validation_rows
    ranked = frame.groupby("candidate").agg(sse=("sse", "sum"), rows=("validation_rows", "sum")).reset_index()
    ranked["pooled_rmse"] = np.sqrt(ranked.sse / ranked.rows)
    return ranked.sort_values(["pooled_rmse", "candidate"]).reset_index(drop=True)


class Experiment:
    def __init__(self, source: Path, output: Path, threads: int = 4):
        import catboost
        import lightgbm
        import sklearn
        self.output, self.threads = output.resolve(), threads
        self.output.mkdir(parents=True, exist_ok=True)
        code = [Path(__file__), ROOT / "src/features/build_temporal_features.py",
            ROOT / "src/models/run_stage5.py", ROOT / "src/models/run_tree_tuning.py",
            ROOT / "src/features/build_stage5_features.py", ROOT / "src/models/train_baseline.py"]
        self.manifest = dict(source=str(source.relative_to(ROOT)), source_sha256=sha(source),
            implementation_sha256={str(p.relative_to(ROOT)): sha(p) for p in code},
            environment=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                sklearn=sklearn.__version__, catboost=catboost.__version__, lightgbm=lightgbm.__version__),
            threads=threads, seed=42, splits=SPLITS, primary_used_for_selection=False,
            primary_previously_inspected=True, test_data_read=False, submission_generated=False,
            selection_rule="sequential tuning on pooled Jul-Sep/Oct-Dec 2024 RMSE; incumbent always eligible",
            cap_rule="one shared inner 2400-cap trajectory; best prefix <=800/1600/2400; refit each distinct selected round count",
            recency_screen="LightGBM screens all four recipes; confirm best new recipe with CatBoost",
            history_screen="LightGBM screens 6/12-month histories; confirm best new recipe with CatBoost",
            trend_rule="static LightGBM residual + flat3/damped12/damped24 monthly trend; blend weights 0/25/50/75/100%",
            long_backtests_use="diagnostic only, not additional selection; earlier training origins and overlapping 2024 queries",
            limits=["static HDB snapshot availability", "repeatedly inspected primary quarter", "bounded screening and greedy selection", "monthly trend is affected by housing composition"])
        self.run_id = signature(self.manifest)
        self.cache = ROOT / "models/rmse_improvement_cache" / self.run_id
        self.cache.mkdir(parents=True, exist_ok=True)
        existing = self.output / "experiment_manifest.json"
        if existing.exists() and json.loads(existing.read_text()).get("run_id") != self.run_id:
            raise ValueError("Output belongs to another experiment; choose a new --output-dir")
        self.manifest["run_id"] = self.run_id
        dump(existing, self.manifest)
        self.train = load_training(source)
        self.splits = {}
        months = self.train.RENT_APPROVAL_DATE.dt.to_period("M")
        for name, (start, end) in SPLITS.items():
            first, last = pd.Period(start), pd.Period(end)
            self.splits[name] = (self.train.loc[months.lt(first)].copy(), self.train.loc[months.between(first, last)].copy())
        self.specs: dict[str, Spec] = {}
        self.choices: dict[str, dict[str, float]] = {}
        self.predictions: dict[tuple[str, str], np.ndarray] = {}
        self.rows: list[dict] = []
        self.stages: list[dict] = []

    def _fit(self, fit: pd.DataFrame, query: pd.DataFrame, spec: Spec, rounds: int | None):
        early = rounds is None
        recipe = asdict(spec)
        # cap only restricts selected prefix; all CatBoost inner runs share the
        # same predeclared 2400-round ceiling and 80-round patience.
        recipe.pop("cap")
        key = signature(dict(recipe=recipe, fit_start=str(fit.RENT_APPROVAL_DATE.min()),
            fit_end=str(fit.RENT_APPROVAL_DATE.max()), rows=len(fit), rounds=rounds, early=early))
        path = self.cache / f"fit_{key}.joblib"
        if path.exists():
            return joblib.load(path)
        started = time.perf_counter()
        train, valid, cats, nums, trend = prepare(fit, query, spec)
        levels = {c: train[c].fillna("missing").astype(str).unique().tolist() for c in cats}
        bundle = FittedModel(spec.family, None, cats, nums, levels)
        weights = sample_weights(fit, spec)
        grid = search_grid()
        params = grid["catboost_d8_regularized" if spec.family == "catboost" else "lightgbm_15_regularized"]["params"]
        count = (2400 if spec.family == "catboost" else 1000) if early else rounds
        if spec.family == "catboost":
            from catboost import CatBoostRegressor
            model = CatBoostRegressor(iterations=count, **params, loss_function="RMSE", random_seed=42,
                thread_count=self.threads, allow_writing_files=False, verbose=False)
            kwargs = dict(cat_features=cats, sample_weight=weights)
            if early:
                labels = query[TARGET].to_numpy()
                if trend is not None:
                    labels = labels - trend_values(trend, query)
                kwargs.update(eval_set=(bundle.matrix(valid), labels), early_stopping_rounds=80, use_best_model=True)
        else:
            from lightgbm import LGBMRegressor, early_stopping
            model = LGBMRegressor(n_estimators=count, **params, max_depth=-1, objective="regression",
                random_state=42, n_jobs=self.threads, verbosity=-1, deterministic=True, force_col_wise=True)
            kwargs = dict(categorical_feature=cats, sample_weight=weights)
            if early:
                labels = query[TARGET].to_numpy()
                if trend is not None:
                    labels = labels - trend_values(trend, query)
                kwargs.update(eval_set=[(bundle.matrix(valid), labels)], eval_metric="rmse", callbacks=[early_stopping(80, verbose=False)])
        model.fit(bundle.matrix(train), train[TARGET], **kwargs)
        bundle.model = model
        history = model.get_evals_result()["validation"]["RMSE"] if early and spec.family == "catboost" else []
        chosen = (int(model.tree_count_) if spec.family == "catboost" else int(model.best_iteration_)) if early else rounds
        payload = dict(bundle=bundle, trend=trend, history=history, chosen=chosen,
            seconds=time.perf_counter() - started)
        temporary = path.with_suffix(".tmp")
        joblib.dump(payload, temporary)
        temporary.replace(path)
        print(f"FIT {spec.name} {'inner' if early else 'refit'} through {fit.RENT_APPROVAL_DATE.max():%Y-%m}: rounds={chosen}, {payload['seconds']:.1f}s", flush=True)
        return payload

    def component(self, split: str, spec: Spec) -> np.ndarray:
        if (split, spec.name) in self.predictions:
            return self.predictions[(split, spec.name)]
        self.specs[spec.name] = spec
        path = self.cache / f"pred_{split}_{spec.name}.npz"
        meta = path.with_suffix(".json")
        fit, query = self.splits[split]
        started = time.perf_counter()
        if path.exists() and meta.exists():
            with np.load(path) as saved:
                prediction = saved["prediction"]
                if not np.array_equal(saved["source_row"], query.index):
                    raise ValueError("Cached query row order mismatch")
            details = json.loads(meta.read_text())
        else:
            selected = select_training(fit, spec)
            inner_fit, inner_valid = split_inner(selected)
            tuned = self._fit(inner_fit, inner_valid, spec, None)
            if spec.family == "catboost":
                losses = np.asarray(tuned["history"][:spec.cap])
                rounds = int(np.argmin(losses)) + 1
            else:
                rounds = max(1, tuned["chosen"])
            payload = self._fit(selected, query, spec, rounds)
            _, featured, _, _, _ = prepare(selected, query, spec)
            prediction = np.asarray(payload["bundle"].model.predict(payload["bundle"].matrix(featured)), dtype=float)
            if payload["trend"] is not None:
                prediction += trend_values(payload["trend"], query)
            prediction = np.maximum(prediction, 0)
            details = dict(iterations=rounds, cap=spec.cap if spec.family == "catboost" else 1000,
                train_rows=len(selected), effective_train_start=f"{selected.RENT_APPROVAL_DATE.min():%Y-%m}",
                inner_fit_end=f"{inner_fit.RENT_APPROVAL_DATE.max():%Y-%m}",
                inner_validation_start=f"{inner_valid.RENT_APPROVAL_DATE.min():%Y-%m}",
                inner_validation_end=f"{inner_valid.RENT_APPROVAL_DATE.max():%Y-%m}",
                seconds=time.perf_counter() - started, spec=asdict(spec))
            np.savez_compressed(path, prediction=prediction, source_row=query.index.to_numpy())
            dump(meta, details)
        if len(prediction) != len(query) or not np.isfinite(prediction).all():
            raise ValueError("Invalid predictions")
        self.predictions[(split, spec.name)] = prediction
        dump(self.output / f"fit_{split}_{spec.name}.json", details)
        return prediction

    def register(self, name: str, weights: dict[str, float]) -> str:
        if not weights or min(weights.values()) < 0 or not np.isclose(sum(weights.values()), 1):
            raise ValueError("Invalid blending weights")
        self.choices[name] = weights
        return name

    def pair(self, name: str, cat: Spec, light: Spec) -> str:
        self.specs[cat.name], self.specs[light.name] = cat, light
        return self.register(name, {cat.name: .5, light.name: .5})

    def predict(self, split: str, name: str) -> np.ndarray:
        weights = self.choices[name]
        # At most two fits run at a time; each is limited to --threads cores.
        with ThreadPoolExecutor(max_workers=min(2, len(weights))) as pool:
            pairs = list(weights.items())
            values = list(pool.map(lambda pair: self.component(split, self.specs[pair[0]]), pairs))
        return sum(value * weight for value, (_, weight) in zip(values, pairs))

    def evaluate(self, names: list[str], splits: list[str], stage: str):
        for split in splits:
            fit, query = self.splits[split]
            for name in names:
                if any(row["candidate"] == name and row["split"] == split for row in self.rows):
                    continue
                prediction = self.predict(split, name)
                self.rows.append(dict(**split_metadata(split, fit, query), candidate=name, stage=stage,
                    **metrics(query[TARGET], prediction)))
                pd.DataFrame(self.rows).to_csv(self.output / "results.csv", index=False)
                print(f"RESULT {stage} {split} {name}: RMSE={self.rows[-1]['rmse']:.4f}", flush=True)

    def choose(self, stage: str, names: list[str], incumbent: str | None = None) -> str:
        self.evaluate(names, EARLIER, stage)
        ranked = score_choice(self.rows, names)
        selected = str(ranked.iloc[0].candidate)
        ranked.to_csv(self.output / f"selection_{stage}.csv", index=False)
        record = dict(stage=stage, selected=selected, pooled_rmse=float(ranked.iloc[0].pooled_rmse),
            incumbent=incumbent, candidates=names, primary_used=False)
        self.stages.append(record)
        dump(self.output / "stage_decisions.json", self.stages)
        print(f"LOCK {stage}: {selected}, pooled={record['pooled_rmse']:.4f}", flush=True)
        return selected

    def checkpoint(self, stage: str, incumbent: str):
        dump(self.output / "progress.json", dict(last_completed_stage=stage, current_choice=incumbent,
            choice_weights=self.choices[incumbent], specs={k: asdict(v) for k, v in self.specs.items()},
            decisions=self.stages))

    def run(self, stop_after: str = "all"):
        cat, light = Spec("catboost", 800), Spec("lightgbm", 1000)
        baseline = self.pair("control_800_blend", cat, light)
        caps = [baseline] + [self.pair(f"cap_{cap}_blend", replace(cat, cap=cap), light) for cap in [1600, 2400]]
        incumbent = self.choose("01_caps", caps, baseline)
        cat = self.specs[next(k for k in self.choices[incumbent] if k.startswith("catboost"))]
        checkpoints = [baseline, incumbent]
        self.checkpoint("01_caps", incumbent)
        if stop_after == "caps":
            return

        recipes = [dict(window=24), dict(window=36), dict(half_life=12), dict(half_life=24)]
        recency_names = []
        for recipe in recipes:
            spec = replace(light, **recipe)
            self.specs[spec.name] = spec
            recency_names.append(self.register(f"recency_screen_{spec.name}", {spec.name: 1.0}))
        base_light = self.register("recency_screen_control", {light.name: 1.0})
        self.evaluate([base_light] + recency_names, EARLIER, "02_recency_screen")
        ranking = score_choice(self.rows, recency_names)
        screened = self.specs[next(iter(self.choices[str(ranking.iloc[0].candidate)]))]
        challenger_cat = replace(cat, window=screened.window, half_life=screened.half_life)
        challenger = self.pair("recency_confirmed_blend", challenger_cat, screened)
        selected = self.choose("02_recency", [incumbent, challenger], incumbent)
        if selected == challenger:
            cat, light = challenger_cat, screened
        incumbent = selected
        checkpoints.append(incumbent)
        self.checkpoint("02_recency", incumbent)

        history_names = []
        for window in [6, 12]:
            spec = replace(light, history=window)
            self.specs[spec.name] = spec
            history_names.append(self.register(f"history_screen_{window}", {spec.name: 1.0}))
        self.evaluate(history_names, EARLIER, "03_history_screen")
        ranking = score_choice(self.rows, history_names)
        screened = self.specs[next(iter(self.choices[str(ranking.iloc[0].candidate)]))]
        challenger_cat = replace(cat, history=screened.history)
        challenger = self.pair("history_confirmed_blend", challenger_cat, screened)
        selected = self.choose("03_history", [incumbent, challenger], incumbent)
        if selected == challenger:
            cat, light = challenger_cat, screened
        incumbent = selected
        checkpoints.append(incumbent)
        self.checkpoint("03_history", incumbent)

        trend_names = [incumbent]
        for kind in ["flat3", "damped12", "damped24"]:
            spec = replace(light, history=0, trend=kind)
            self.specs[spec.name] = spec
            for weight in [.25, .5, .75, 1.0]:
                weights = {k: v * (1 - weight) for k, v in self.choices[incumbent].items() if weight < 1}
                weights[spec.name] = weight
                name = self.register(f"trend_{kind}_weight{int(weight * 100)}", weights)
                trend_names.append(name)
        incumbent = self.choose("04_trend", trend_names, incumbent)
        checkpoints.append(incumbent)
        self.checkpoint("04_trend", incumbent)
        lock = dict(selected=incumbent, weights=self.choices[incumbent], baseline=baseline,
            specs={k: asdict(v) for k, v in self.specs.items()}, decisions=self.stages,
            primary_used=False, primary_previously_inspected=True)
        dump(self.output / "selection_locked.json", lock)

        # All tuning choices are fixed before accessing these scores.
        self.evaluate(list(dict.fromkeys(checkpoints)), ["primary_2025"], "05_primary")
        self.evaluate(list(dict.fromkeys([baseline, incumbent])), ["long_6m", "long_12m"], "06_long_horizon")
        for split in ["primary_2025", "long_6m", "long_12m"]:
            query = self.splits[split][1]
            preds = {name: self.predict(split, name) for name in dict.fromkeys([baseline, incumbent])}
            pd.DataFrame(dict(source_row=query.index, actual=query[TARGET].to_numpy(), **preds)).to_csv(self.output / f"predictions_{split}.csv", index=False)
            grouped_errors(query, preds).to_csv(self.output / f"group_errors_{split}.csv", index=False)
        for split in EARLIER:
            query = self.splits[split][1]
            preds = {name: self.predict(split, name) for name in self.choices if any(row['candidate'] == name and row['split'] == split for row in self.rows)}
            pd.DataFrame(dict(source_row=query.index, actual=query[TARGET].to_numpy(), **preds)).to_csv(self.output / f"predictions_{split}.csv", index=False)
        self.write_report(baseline, incumbent)
        self.manifest["complete"] = True
        self.manifest["selected"] = incumbent
        dump(self.output / "experiment_manifest.json", self.manifest)
        self.checkpoint("complete", incumbent)
        print(f"DONE {self.output}", flush=True)

    def write_report(self, baseline: str, selected: str):
        rows = pd.DataFrame(self.rows)
        primary = rows.loc[rows.split.eq("primary_2025")]
        before = float(primary.loc[primary.candidate.eq(baseline), "rmse"].iloc[0])
        after = float(primary.loc[primary.candidate.eq(selected), "rmse"].iloc[0])
        lines = ["# 降低 RMSE 的顺序实验", "", "## 实验规则", "",
            "- 以同环境复跑的 800 轮 CatBoost / 调优 LightGBM 等权融合为对照；历史分数 450.4735 只作参考。",
            "- 各阶段仅按 2024 年第三、四季度的合并平方误差选择；保留旧方案作为候选。后续阶段依赖前一阶段选择，属于有界顺序调优，不是全面搜索。",
            "- 近期方案与历史统计先由 LightGBM 筛选，再用 CatBoost 确认一个新方案，以控制计算成本。",
            "- CatBoost 在内部更早的三个月验证上早停；800/1600/2400 上限共享一条 2400 轮上限轨迹，按各上限前的最佳轮数分别重训。",
            "- 训练行的租金统计只使用更早月份；整段验证查询冻结在训练截止时。未按同月其他行或未来验证租金更新。",
            "- 趋势实验将训练月份的整体租金水平移除，以静态房屋特征拟合残差；未来水平使用最近三个月常数或最近 12/24 个月的六个月阻尼趋势。",
            "- 选定后才报告 2025 年第一季度及 6/12 个月窗口。主验证以前已被查看；长窗口与调优季度存在重叠，只是不同训练起点的诊断。",
            "- 不读取竞赛测试集、不生成提交、不删除重复观测或极端租金。", "", "## 阶段选择", "",
            "| 阶段 | 选定方案 | 较早季度合并 RMSE |", "| --- | --- | ---: |"]
        for stage in self.stages:
            lines.append(f"| {stage['stage']} | {stage['selected']} | {stage['pooled_rmse']:.4f} |")
        lines += ["", "## 主时间验证：2025 年 1–3 月", "", "| 方案 | RMSE | MAE | 平均偏差 |", "| --- | ---: | ---: | ---: |"]
        for row in primary.itertuples():
            lines.append(f"| {row.candidate} | {row.rmse:.4f} | {row.mae:.4f} | {row.bias:+.2f} |")
        lines += ["", f"本地对照 RMSE {before:.4f}，预先锁定方案 {after:.4f}；降低 {before-after:+.4f}（{(before-after)/before:+.2%}）。负值表示新方案在主验证变差，仍保留原先选择，不按主验证事后重选。",
            "", "## 长跨度诊断", "", "| 窗口 | 方案 | RMSE | MAE | 偏差 |", "| --- | --- | ---: | ---: | ---: |"]
        for row in rows.loc[rows.split.str.startswith("long_")].itertuples():
            lines.append(f"| {row.split} | {row.candidate} | {row.rmse:.4f} | {row.mae:.4f} | {row.bias:+.2f} |")
        lines += ["", "## 局限与产物", "",
            "- 租金统计不识别单个住房；同楼栋同房型仍可有不同真实租金。窗口开始处无历史的行保留空值，不能用未来均值填补。",
            "- 月度平均租金可能受房型和区位构成变化影响，趋势分解不是独立市场价格指数。",
            "- 只尝试预设窗口、权重和融合比例；未试完所有组合，也不保证排行榜表现。静态 HDB 信息仍有历史可用性限制。",
            "- `results.csv` 与各 `selection_*.csv` 为汇总分数；`stage_decisions.json`、`selection_locked.json` 记录选择。",
            "- `predictions_*.csv`、`group_errors_*.csv` 和 `models/rmse_improvement_cache/` 为本地诊断产物；逐行租金/预测和模型不进入 Git。",
            "- 配置、输入及实现 SHA-256 与依赖版本见 `experiment_manifest.json`；相同配置中断重跑可恢复。", ""]
        (self.output / "improvement_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/processed/train_location_features.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/experiments/rmse_improvement_v1")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--stop-after", choices=["caps", "all"], default="all")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    Experiment(args.source.resolve(), args.output_dir, args.threads).run(args.stop_after)
