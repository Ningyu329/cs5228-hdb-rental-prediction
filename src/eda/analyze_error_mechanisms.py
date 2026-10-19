"""Read-only model diagnosis: tails, past coverage and identical-input spread.

No model is fitted or changed; validation labels are used for diagnosis only.
The competition test table and submitted predictions are never read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TARGET = "MONTHLY_RENT"
DATE = "RENT_APPROVAL_DATE"
MODEL = "locked_448_control"
INPUTS = [DATE, "TOWN", "BLOCK", "STREET", "FLAT_TYPE", "FLAT_MODEL", "FLOOR_AREA_SQM",
    "FURNISHED", "LEASE_COMMENCE_DATE", "FEE"]
EXACT_HOUSE = ["TOWN", "BLOCK", "STREET", "FLAT_TYPE", "FLAT_MODEL", "FLOOR_AREA_SQM", "LEASE_COMMENCE_DATE"]
WINDOWS = {"q3_2024": ("2024-07", "2024-09"), "q4_2024": ("2024-10", "2024-12"), "primary_2025": ("2025-01", "2025-03")}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collisions(frame: pd.DataFrame, columns: list[str]) -> dict:
    """Retrospective finite-sample bound, NOT an estimated Bayes error floor."""
    group = frame.groupby(columns, dropna=False, observed=True)[TARGET]
    means, counts = group.transform("mean"), group.transform("size")
    within = float(((frame[TARGET] - means) ** 2).sum())
    summary = group.agg(["size", "nunique", "min", "max"])
    duplicate = summary.loc[summary["size"].gt(1)]
    conflict = summary.loc[summary["nunique"].gt(1)]
    return dict(rows=len(frame), groups=len(summary), repeated_groups=len(duplicate),
        repeated_rows=int(duplicate["size"].sum()), conflicting_groups=len(conflict),
        conflicting_rows=int(conflict["size"].sum()), within_sse=within,
        retrospective_constant_rmse=float(np.sqrt(within / len(frame))),
        repeated_rows_rmse=float(np.sqrt(within / counts.gt(1).sum())) if counts.gt(1).any() else None,
        max_conflicting_rent_range=float((conflict["max"] - conflict["min"]).max()) if len(conflict) else None)


def attach_history(fit: pd.DataFrame, query: pd.DataFrame) -> pd.DataFrame:
    """Frozen past covariates for diagnosis, independent of query rental labels."""
    fit, query = fit.copy(), query.copy()
    fit[DATE], query[DATE] = pd.to_datetime(fit[DATE]), pd.to_datetime(query[DATE])
    if fit.empty or fit[DATE].max() >= query[DATE].min():
        raise ValueError("Historical labels must precede the whole query window")
    last = fit[DATE].max().to_period("M")
    recent = fit.loc[fit[DATE].dt.to_period("M").gt(last - 12)].copy()
    # Used only as a historical descriptive spread. A future training feature
    # would need expanding-origin/OOF construction on every training row too.
    recent["_ADJUSTED_RENT"] = recent[TARGET] - recent.groupby([DATE, "FLAT_TYPE"])[TARGET].transform("mean")
    blocks = ["TOWN", "BLOCK", "STREET", "FLAT_TYPE"]
    lookup = pd.MultiIndex.from_frame(query[blocks])
    grouped = recent.groupby(blocks, dropna=False).agg(
        n=(TARGET, "size"), mean=(TARGET, "mean"), median=(TARGET, "median"),
        std=(TARGET, "std"), adjusted_std=("_ADJUSTED_RENT", "std"),
        q10=(TARGET, lambda x: x.quantile(.1)), q90=(TARGET, lambda x: x.quantile(.9)))
    joined = grouped.reindex(lookup)
    for name in joined:
        query[f"HISTORY_{name.upper()}"] = joined[name].to_numpy()
    query["HISTORY_N"] = query.HISTORY_N.fillna(0)
    full = fit.groupby(blocks, dropna=False)[TARGET].size().reindex(lookup).fillna(0)
    query["ALL_HISTORY_N"] = full.to_numpy()
    exact = fit.groupby(EXACT_HOUSE, dropna=False)[TARGET].size()
    query["EXACT_HOUSE_N"] = exact.reindex(pd.MultiIndex.from_frame(query[EXACT_HOUSE])).fillna(0).to_numpy()
    return query


def summarize_partition(view: pd.DataFrame, field: str) -> pd.DataFrame:
    rows, total = [], float(view.SQUARED_ERROR.sum())
    for value, group in view.groupby(field, dropna=False, observed=True):
        e = group.ERROR.to_numpy()
        rows.append(dict(partition=field, segment=str(value), rows=len(group), row_share=len(group)/len(view),
            rmse=float(np.sqrt(np.mean(e**2))), mae=float(np.abs(e).mean()), bias=float(e.mean()),
            mean_actual=float(group[TARGET].mean()), mean_prediction=float(group.prediction.mean()),
            sse=float(np.dot(e,e)), sse_share=float(np.dot(e,e)/total), tail_rate=float(group.TAIL5.mean()),
            errors_over_1000=int((np.abs(e)>1000).sum()), over_1000=int((e>1000).sum()), under_1000=int((e < -1000).sum())))
    return pd.DataFrame(rows).sort_values("sse", ascending=False)


def make_view(fit: pd.DataFrame, query: pd.DataFrame, prediction: np.ndarray) -> pd.DataFrame:
    view = attach_history(fit, query)
    view["prediction"] = prediction
    view["ERROR"] = prediction - view[TARGET]
    view["SQUARED_ERROR"] = view.ERROR**2
    view["TAIL5"] = view.ERROR.abs().rank(method="first", ascending=False).le(int(np.ceil(len(view)*.05)))
    bins = [-np.inf, 1500, 2000, 2500, 3000, 3500, 4000, 4500, np.inf]
    view["ACTUAL_RENT_BIN"] = pd.cut(view[TARGET], bins).astype(str)
    view["PREDICTED_RENT_BIN"] = pd.cut(view.prediction, bins).astype(str)
    view["RECENT_HISTORY_BIN"] = pd.cut(view.HISTORY_N, [-1,0,4,9,19,np.inf], labels=["unseen","1-4","5-9","10-19","20+"]).astype(str)
    view["ALL_HISTORY_BIN"] = pd.cut(view.ALL_HISTORY_N, [-1,0,4,19,49,np.inf], labels=["unseen","1-4","5-19","20-49","50+"]).astype(str)
    view["EXACT_HOUSE_BIN"] = pd.cut(view.EXACT_HOUSE_N, [-1,0,4,19,np.inf], labels=["unseen","1-4","5-19","20+"]).astype(str)
    spread = pd.cut(view.HISTORY_ADJUSTED_STD, [-np.inf,200,350,500,np.inf], labels=["<=200","200-350","350-500",">500"]).astype("string").fillna("unavailable")
    view["HISTORICAL_SPREAD_BIN"] = spread.where(view.HISTORY_N.ge(5), "fewer than 5")
    view["RENT_BELOW_HIST_Q10"] = np.where(view.HISTORY_N.ge(10), np.where(view[TARGET].lt(view.HISTORY_Q10), "below historical q10", "not below historical q10"), "fewer than 10")
    view["AREA_BIN"] = pd.cut(view.FLOOR_AREA_SQM, [-np.inf,50,70,90,110,130,160,np.inf]).astype(str)
    view["AGE_BIN"] = pd.cut(view.FLAT_AGE, [-np.inf,10,20,30,40,50,np.inf]).astype("string").fillna("missing")
    return view


def run(output: Path) -> dict:
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a fresh diagnostic output directory")
    output.mkdir(parents=True, exist_ok=True)
    source = ROOT / "data/processed/train_location_features.csv"
    raw_path = ROOT / "data/raw/train.csv"
    train, raw = pd.read_csv(source), pd.read_csv(raw_path)
    if len(train) != len(raw):
        raise ValueError("Raw/processed row counts differ")
    np.testing.assert_array_equal(train[TARGET], raw[TARGET])
    # Verify all normalized input columns in exact source row order as well.
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from src.data.clean_data import read_csv, clean_main_rows
    original, _ = read_csv(raw_path)
    cleaned, fields = clean_main_rows(original, has_target=True)
    expected = pd.DataFrame(cleaned)
    for col in fields:
        if col in {DATE,"TOWN","BLOCK","STREET","FLAT_TYPE","FLAT_MODEL","FURNISHED"}:
            np.testing.assert_array_equal(train[col].fillna("").astype(str), expected[col])
        else:
            np.testing.assert_allclose(pd.to_numeric(train[col]), pd.to_numeric(expected[col].replace("", np.nan)), rtol=0, atol=1e-9, equal_nan=True)
    train[DATE] = pd.to_datetime(train[DATE])
    raw_style = raw.FLAT_TYPE.str.contains(" room", regex=False)
    formatting = raw.assign(style=np.where(raw_style,"space","hyphen_or_executive")).groupby("style").agg(
        rows=(TARGET,"size"),first_month=(DATE,"min"),last_month=(DATE,"max"),mean_rent=(TARGET,"mean"))
    formatting.to_csv(output / "raw_formatting.csv")
    source_paths = [source, raw_path, Path(__file__)]
    summary, all_groups = {}, []
    for split, (start,end) in WINDOWS.items():
        path = ROOT / f"reports/experiments/segment_v1/predictions_{split}.csv"
        source_paths.append(path)
        stored = pd.read_csv(path)
        months = train[DATE].dt.to_period("M")
        fit = train.loc[months.lt(pd.Period(start))].copy()
        query = train.loc[months.between(pd.Period(start),pd.Period(end))].copy()
        np.testing.assert_array_equal(stored.source_row, query.index)
        np.testing.assert_array_equal(stored.actual, query[TARGET])
        query["RAW_FLAT_TYPE"] = raw.iloc[query.index].FLAT_TYPE.to_numpy()
        view = make_view(fit, query, stored[MODEL].to_numpy())
        partitions = ["ACTUAL_RENT_BIN","PREDICTED_RENT_BIN","RECENT_HISTORY_BIN","ALL_HISTORY_BIN",
            "EXACT_HOUSE_BIN","HISTORICAL_SPREAD_BIN","RENT_BELOW_HIST_Q10","FLAT_TYPE","TOWN","RAW_FLAT_TYPE","AREA_BIN","AGE_BIN"]
        tables = []
        for field in partitions:
            table = summarize_partition(view,field)
            assert table.rows.sum() == len(view)
            np.testing.assert_allclose(table.sse_share.sum(),1,atol=1e-10)
            table["split"] = split
            tables.append(table)
        table = pd.concat(tables,ignore_index=True)
        table.to_csv(output / f"error_mechanisms_{split}.csv",index=False)
        all_groups.append(table)
        errors = view.SQUARED_ERROR.to_numpy()
        descending = np.sort(errors)[::-1]
        tails = {}
        for fraction in [.01,.05,.10,.20]:
            top = float(descending[:int(np.ceil(len(view)*fraction))].sum())
            tails[str(fraction)] = dict(sse_share=top/float(errors.sum()),
                hypothetical_rmse_if_tail_errors_zero=float(np.sqrt(max(0,float(errors.sum())-top)/len(view))))
        tail = view.loc[view.TAIL5]
        repeated = collisions(query,INPUTS)
        raw_repeated = collisions(raw.iloc[query.index],INPUTS)
        reliable = view.HISTORY_N.ge(5) & view.HISTORY_ADJUSTED_STD.notna()
        summary[split] = dict(rows=len(view),rmse=float(np.sqrt(errors.mean())),
            mae=float(view.ERROR.abs().mean()),bias=float(view.ERROR.mean()),
            actual_std=float(view[TARGET].std()),prediction_std=float(view.prediction.std()),
            error_over_1000=int(view.ERROR.gt(1000).sum()),error_under_minus1000=int(view.ERROR.lt(-1000).sum()),
            actual_below_2000_sse_share=float(view.loc[view[TARGET].le(2000),"SQUARED_ERROR"].sum()/errors.sum()),
            actual_below_2000_row_share=float(view[TARGET].le(2000).mean()),
            tail5_actual_mean=float(tail[TARGET].mean()),tail5_prediction_mean=float(tail.prediction.mean()),
            tail5_known_exact_house_share=float(tail.EXACT_HOUSE_N.gt(0).mean()),
            all_known_exact_house_share=float(view.EXACT_HOUSE_N.gt(0).mean()),
            reliable_historical_spread_coverage=float(reliable.mean()),
            historical_spread_absolute_error_spearman=float(view.loc[reliable,"HISTORY_ADJUSTED_STD"].rank().corr(view.loc[reliable,"ERROR"].abs().rank())) if reliable.any() else None,
            normalized_identical_inputs=repeated,raw_identical_inputs=raw_repeated,tails=tails)
        # Sensitive diagnostic rows remain local and excluded from Git.
        columns = [DATE,"TOWN","BLOCK","STREET","FLAT_TYPE","FLAT_MODEL","FLOOR_AREA_SQM","FLAT_AGE",TARGET,"prediction","ERROR","HISTORY_N","HISTORY_STD","HISTORY_ADJUSTED_STD","EXACT_HOUSE_N"]
        view.loc[view.TAIL5,columns].sort_values("ERROR").to_csv(output / f"tail_samples_{split}.csv",index=False)
        print(json.dumps({split:summary[split]},ensure_ascii=False,indent=2),flush=True)
    combined = pd.concat(all_groups,ignore_index=True)
    combined.to_csv(output / "error_mechanisms.csv",index=False)
    manifest = dict(source_sha256={str(p.relative_to(ROOT)):sha(p) for p in source_paths},
        python=platform.python_version(),pandas=pd.__version__,numpy=np.__version__,
        test_data_read=False,models_changed=False,rows_removed=False,
        actual_rent_bins_use="retrospective diagnosis only; not deployable routing features",
        constant_fields={c:int(train[c].nunique(dropna=False)) for c in ["FURNISHED","FEE"]},
        retrospective_collision_bound="finite observed sample only; singleton groups contribute zero, not a Bayes error estimate",
        history_rule="12-month frozen block+type history; exact-house counts use all prior training rows",
        summary=summary)
    (output/"diagnosis.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    write_report(output,manifest,combined,formatting)
    return manifest


def write_report(output: Path, manifest: dict, combined: pd.DataFrame, formatting: pd.DataFrame):
    lines = ["# 大误差成因诊断", "", "本轮只分析当前已提交模型的本地验证误差，不训练或修改租金模型、不删除样本、不读取竞赛测试集。", "",
        "## 误差方向与租金水平", "", "| 窗口 | RMSE | 高估超过 1000 | 低估超过 1000 | 实际租金 ≤2000 的样本占比 | 其 SSE 占比 |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for split,row in manifest["summary"].items():
        lines.append(f"| {split} | {row['rmse']:.2f} | {row['error_over_1000']} | {row['error_under_minus1000']} | {row['actual_below_2000_row_share']:.1%} | {row['actual_below_2000_sse_share']:.1%} |")
    lines += ["", "按实际租金分组会天然出现低租金高估、高租金低估，即使条件均值模型合理也可能如此。因此这只是误差集中位置，不能据此直接给未知低租金测试样本降价。", "",
        "## 历史可比房屋覆盖", "", "可比条件为城镇、楼栋、街道、房型、房屋型号、面积、租约起始年，不包含单元号；历史窗口严格早于整个验证窗口。", "",
        "| 窗口 | 全体验证样本有历史可比记录 | 最差 5% 有历史可比记录 | 有至少 5 条近期楼栋房型记录 | 历史离散度与绝对误差 Spearman 相关 |", "| --- | ---: | ---: | ---: | ---: |"]
    for split,row in manifest["summary"].items():
        lines.append(f"| {split} | {row['all_known_exact_house_share']:.1%} | {row['tail5_known_exact_house_share']:.1%} | {row['reliable_historical_spread_coverage']:.1%} | {row['historical_spread_absolute_error_spearman']:.3f} |")
    lines += ["", "历史离散度在过去十二个月的同楼栋同房型内计算，并先减去各历史月同房型的平均租金。相关性不代表因果；样本稀少时估计不可靠，未来若用作训练特征，必须对每条训练记录另做过去信息/OOF 构造。", "",
        "## 完全相同可观测输入仍有多少租金差异", "", "| 窗口 | 重复输入涉及的样本数 | 租金冲突的输入组数 | 事后组均值的全样本 RMSE | 原始未规范化输入的同类 RMSE |", "| --- | ---: | ---: | ---: | ---: |"]
    for split,row in manifest["summary"].items():
        col,raw=row["normalized_identical_inputs"],row["raw_identical_inputs"]
        lines.append(f"| {split} | {col['repeated_rows']} | {col['conflicting_groups']} | {col['retrospective_constant_rmse']:.2f} | {raw['retrospective_constant_rmse']:.2f} |")
    lines += ["", "此处用验证集自身的标签计算每个完整输入组的事后均值，仅用于描述。它是当前有限样本、给定输入下确定性预测的经验约束，不是可部署预测，也不是新样本不可约误差估计。单例组在此统计中误差为零，不能因此推断它们可预测。", "",
        "## 尾部误差的影响", "", "| 窗口 | 最差 5% SSE 占比 | 假设该 5% 误差全部归零后的整体 RMSE | 最差 10% SSE 占比 | 假设该 10% 误差归零后的整体 RMSE |", "| --- | ---: | ---: | ---: | ---: |"]
    for split,row in manifest["summary"].items():
        a,b=row["tails"]["0.05"],row["tails"]["0.1"]
        lines.append(f"| {split} | {a['sse_share']:.1%} | {a['hypothetical_rmse_if_tail_errors_zero']:.2f} | {b['sse_share']:.1%} | {b['hypothetical_rmse_if_tail_errors_zero']:.2f} |")
    lines += ["", "上表保留全体样本作分母，不是删除难例后的分数。它依赖事后已知误差，不能作为实际可实现的提升承诺。", "",
        "## 清洗与原始格式复查", "",
        "- 已将规范化后的全部主输入列与原始训练表按行重新计算核对，训练标签及行序一致。",
        f"- FURNISHED 与 FEE 各只有 {manifest['constant_fields']['FURNISHED']}、{manifest['constant_fields']['FEE']} 个不同取值，无法提供样本间区分信息。",
        f"- 空格形式房型共 {int(formatting.loc['space','rows'])} 条，只出现在 {formatting.loc['space','first_month']} 至 {formatting.loc['space','last_month']}；2024/2025 验证样本没有该形式。原始房型均值差异与年份、房型结构混杂，不能单凭均值认定合并格式损害模型。", "",
        "## 使用限制与下一步", "",
        "- 真实租金分箱、尾部样本及冲突组属于标签已知后的诊断，不可直接用于测试集路由、删样本或标记廉价房屋。",
        "- 历史覆盖、过去租金分布和区位邻域可在预测时获得；是否改善需按时间回测实际验证，不能由相关性保证。",
        "- 若缺少单元号、实际楼层、装修/租约条件等信息，它们可能解释部分同组差异，但本次数据不能验证具体原因，不能把所有残差归为噪声。",
        "- 优先评估过去租金分布/近邻特征是否能区分同区位内部差异；同时审查大误差是否集中在历史便宜租赁或特定可观测属性。未经确认不删除低租金记录。",
        "- 所有验证窗口已被多次查看，本报告不能充当独立最终测试。", ""]
    (output/"error_diagnosis.md").write_text("\n".join(lines),encoding="utf-8")


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=ROOT/"reports/experiments/error_diagnosis_v1")
    run(parser.parse_args().output_dir)
