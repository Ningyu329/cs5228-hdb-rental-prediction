"""Compare raw and cleaned CSV datasets and write a Markdown quality report."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from statistics import median


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def missing_count(rows: list[dict[str, str]], fields: list[str]) -> int:
    return sum(not row.get(field, "").strip() for row in rows for field in fields)


def duplicate_rows(rows: list[dict[str, str]], fields: list[str]) -> int:
    seen: set[tuple[str, ...]] = set()
    duplicates = 0
    for row in rows:
        key = tuple(row.get(field, "") for field in fields)
        if key in seen:
            duplicates += 1
        else:
            seen.add(key)
    return duplicates


def summary(path: Path) -> dict[str, object]:
    rows, fields = read_csv(path)
    return {
        "rows": len(rows),
        "fields": fields,
        "missing": missing_count(rows, fields),
        "duplicates": duplicate_rows(rows, fields),
    }


def address_match_rate(main_rows: list[dict[str, str]], hdb_rows: list[dict[str, str]]) -> tuple[int, int]:
    addresses = {row["ADDRESS_KEY"] for row in hdb_rows}
    matches = sum(
        "|".join([row["TOWN"], row["BLOCK"], row["STREET"]]) in addresses
        for row in main_rows
    )
    return matches, len(main_rows)


def render_table(rows: list[list[str]]) -> list[str]:
    columns = len(rows[0])
    lines = [
        "| " + " | ".join(rows[0]) + " |",
        "| " + " | ".join(["---"] * columns) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows[1:])
    return lines


def distribution_rows(
    train_rows: list[dict[str, str]], test_rows: list[dict[str, str]], field: str
) -> list[list[str]]:
    """返回指定类别字段在训练集与测试集中的计数及占比。"""
    train_counts = Counter(row[field] for row in train_rows)
    test_counts = Counter(row[field] for row in test_rows)
    rows = [[field, "训练集数量", "训练集占比", "测试集数量", "测试集占比", "占比差异（测试-训练）"]]
    for category in sorted(set(train_counts) | set(test_counts)):
        train_share = train_counts[category] / len(train_rows)
        test_share = test_counts[category] / len(test_rows)
        rows.append([
            category,
            f"{train_counts[category]:,}",
            f"{train_share:.1%}",
            f"{test_counts[category]:,}",
            f"{test_share:.1%}",
            f"{test_share - train_share:+.1%}",
        ])
    return rows


def numeric_summary_row(
    train_rows: list[dict[str, str]], test_rows: list[dict[str, str]], field: str
) -> list[str]:
    """返回数值字段在训练集与测试集的最小值、中位数和最大值。"""
    train_values = [float(row[field]) for row in train_rows if row[field]]
    test_values = [float(row[field]) for row in test_rows if row[field]]
    return [
        field,
        f"{min(train_values):.1f}",
        f"{median(train_values):.1f}",
        f"{max(train_values):.1f}",
        f"{min(test_values):.1f}",
        f"{median(test_values):.1f}",
        f"{max(test_values):.1f}",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="生成清洗前后数据质量对比报告。")
    parser.add_argument("--raw-dir", type=Path, default=PROJECT_ROOT / "data" / "raw")
    parser.add_argument("--interim-dir", type=Path, default=PROJECT_ROOT / "data" / "interim")
    parser.add_argument(
        "--report-path",
        type=Path,
        default=PROJECT_ROOT / "reports" / "data_quality_after_cleaning.md",
    )
    args = parser.parse_args()

    raw_dir = args.raw_dir.resolve()
    interim_dir = args.interim_dir.resolve()
    report_path = args.report_path.resolve()

    pairs = [
        ("训练主表", raw_dir / "train.csv", interim_dir / "train_clean.csv"),
        ("测试主表", raw_dir / "test.csv", interim_dir / "test_clean.csv"),
        ("HDB 楼栋表", raw_dir / "auxiliary" / "sg-hdb-block.csv", interim_dir / "auxiliary" / "sg-hdb-block_clean.csv"),
        ("MRT 站点表", raw_dir / "auxiliary" / "sg-mrt-stations.csv", interim_dir / "auxiliary" / "sg-mrt-stations_clean.csv"),
        ("学校表", raw_dir / "auxiliary" / "sg-schools.csv", interim_dir / "auxiliary" / "sg-schools_clean.csv"),
        ("商场表", raw_dir / "auxiliary" / "sg-shopping-malls.csv", interim_dir / "auxiliary" / "sg-shopping-malls_clean.csv"),
        ("COE 表", raw_dir / "auxiliary" / "sg-coe-prices.csv", interim_dir / "auxiliary" / "sg-coe-prices_clean.csv"),
        ("股票表", raw_dir / "auxiliary" / "sg-stock-prices.csv", interim_dir / "auxiliary" / "sg-stock-prices_clean.csv"),
    ]

    report = [
        "# 清洗后数据质量对比报告",
        "",
        "本报告比较 data/raw/ 原始数据与 data/interim/ 清洗结果。清洗过程未删除任何记录，也未改写原始文件。",
        "",
    ]
    table = [["数据集", "原始行数", "清洗后行数", "原始缺失单元格", "清洗后缺失单元格", "原始完整重复行", "清洗后完整重复行"]]
    for label, raw_path, clean_path in pairs:
        raw = summary(raw_path)
        clean = summary(clean_path)
        table.append([
            label,
            f"{raw['rows']:,}",
            f"{clean['rows']:,}",
            f"{raw['missing']:,}",
            f"{clean['missing']:,}",
            f"{raw['duplicates']:,}",
            f"{clean['duplicates']:,}",
        ])
    report.extend(render_table(table))

    raw_train, _ = read_csv(raw_dir / "train.csv")
    clean_train, _ = read_csv(interim_dir / "train_clean.csv")
    raw_test, _ = read_csv(raw_dir / "test.csv")
    clean_test, _ = read_csv(interim_dir / "test_clean.csv")
    raw_hdb, _ = read_csv(raw_dir / "auxiliary" / "sg-hdb-block.csv")
    clean_hdb, _ = read_csv(interim_dir / "auxiliary" / "sg-hdb-block_clean.csv")
    clean_mrt, _ = read_csv(interim_dir / "auxiliary" / "sg-mrt-stations_clean.csv")
    raw_coe, _ = read_csv(raw_dir / "auxiliary" / "sg-coe-prices.csv")
    clean_coe, _ = read_csv(interim_dir / "auxiliary" / "sg-coe-prices_clean.csv")
    clean_stock, _ = read_csv(interim_dir / "auxiliary" / "sg-stock-prices_clean.csv")

    raw_flat_train = len({row["FLAT_TYPE"] for row in raw_train})
    clean_flat_train = len({row["FLAT_TYPE"] for row in clean_train})
    raw_flat_test = len({row["FLAT_TYPE"] for row in raw_test})
    clean_flat_test = len({row["FLAT_TYPE"] for row in clean_test})
    invalid_raw_years = sum(row["YEAR_COMPLETED"] == "-1" for row in raw_hdb)
    blank_clean_years = sum(not row["YEAR_COMPLETED"] for row in clean_hdb)
    train_match, train_total = address_match_rate(clean_train, clean_hdb)
    test_match, test_total = address_match_rate(clean_test, clean_hdb)
    mrt_status = Counter(row["STATUS"] for row in clean_mrt)
    coe_month_numbers = sum(bool(row["MONTH_NUM"]) and bool(row["YEAR_MONTH"]) for row in clean_coe)
    raw_currency_values = sum("$" in row["QUOTA_PREMIUM"] or "," in row["QUOTA_PREMIUM"] for row in raw_coe)
    stock_missing = sum(not row["CLOSE"] for row in clean_stock)
    missing_flat_age = sum(not row["FLAT_AGE"] for row in clean_train)
    town_comparison = distribution_rows(clean_train, clean_test, "TOWN")
    flat_type_comparison = distribution_rows(clean_train, clean_test, "FLAT_TYPE")
    year_comparison = distribution_rows(clean_train, clean_test, "RENT_YEAR")
    numeric_comparison = [
        ["数值字段", "训练集最小值", "训练集中位数", "训练集最大值", "测试集最小值", "测试集中位数", "测试集最大值"],
        numeric_summary_row(clean_train, clean_test, "FLOOR_AREA_SQM"),
        numeric_summary_row(clean_train, clean_test, "FLAT_AGE"),
    ]

    report.extend([
        "",
        "## 1. 已完成的处理",
        "",
        "- 主表新增 RENT_YEAR、RENT_MONTH、RENT_MONTH_INDEX 与 FLAT_AGE，以支持后续基线模型和时间验证。",
        f"- FLAT_TYPE 已标准化：训练集类别数由 {raw_flat_train} 个降至 {clean_flat_train} 个；测试集由 {raw_flat_test} 个保持为 {clean_flat_test} 个。3 room 与 3-room 等变体已统一。",
        "- 主表中的城镇、楼号、街道、房屋模型和装修状态已统一为小写、去首尾空格并压缩连续空格。",
        "- 主表完整重复行数量在清洗后上升，是因为原本仅格式不同但语义相同的记录被标准化为相同值；清洗过程没有删除或合并任何记录。",
        f"- 训练集中有 {missing_flat_age} 条记录的租赁年份早于租约起始年份；其 FLAT_AGE 已保留为空，避免产生不合理的负房龄。",
        f"- HDB 楼栋表中 {invalid_raw_years} 个 YEAR_COMPLETED=-1 已转换为缺失值；清洗后该字段有 {blank_clean_years} 个缺失值。",
        f"- COE 的 {raw_currency_values} 个带货币符号或逗号的溢价字段已转换为数值；{coe_month_numbers:,}/{len(clean_coe):,} 行已补充 MONTH_NUM 与 YEAR_MONTH。",
        "- MRT、学校、商场和股票数据已统一文本、日期与数值格式，但没有进行聚合或删除。",
        "",
        "## 2. 一致性与可用性检查",
        "",
        f"- 清洗后训练集地址匹配：{train_match:,}/{train_total:,}（{train_match / train_total:.1%}）。",
        f"- 清洗后测试集地址匹配：{test_match:,}/{test_total:,}（{test_match / test_total:.1%}）。",
        f"- 清洗后 MRT 状态：已开放 {mrt_status['open']} 个，规划中 {mrt_status['planned']} 个。首版位置特征仅应使用已开放站点。",
        f"- 股票表仍保留 {stock_missing} 行缺失收盘价；这是原始缺失，尚未填补，后续按证券和月份聚合时再定义处理规则。",
        "",
        "## 3. 保留的限制与下一步",
        "",
        "- FEE 全为 0，FURNISHED 全为 yes；两列保留在清洗表中以维持数据谱系，但首版模型应排除。",
        "- 清洗未删除重复记录。训练集中相同输入特征可对应不同租金标签，按特征去重会丢失有效观测。",
        "- COE 同年月同类别可能有多轮记录；本阶段只规范格式，后续特征工程需明确选择最后一轮、均值或其他聚合规则。",
        "- 股票表在本阶段只完成日期与数值字段清洗；月度收益率、波动率等聚合指标将在阶段 5 根据验证结果决定是否构造和接入模型。",
        "- 规划 MRT 站点可能带来时间泄漏；在有明确历史开放时间前，不应纳入早期租赁记录的距离特征。",
        "- 下一步应使用 data/interim/train_clean.csv 和 data/interim/test_clean.csv 建立仅含主表特征的时间切分基线模型。",
        "",
        "## 4. 输入与输出路径",
        "",
        "- 输入：data/raw/，只读保存 Kaggle 原始文件。",
        "- 本阶段输出：data/interim/，保存清洗后的主表和辅助表。",
        "- 后续特征工程输出：data/processed/，保存按实验方案合并后的建模特征表。",
        "- 本报告：reports/data_quality_after_cleaning.md。",
    ])

    report.extend([
        "",
        "## 5. 训练集与测试集分布对比",
        "",
        "下表用于识别测试集与训练集的结构性差异。此处只做数据审计，不分析目标租金；租金分布、时间趋势及其与特征的关系将在阶段 2 的 EDA 中分析。",
        "",
        "### 租赁年份",
        "",
    ])
    report.extend(render_table(year_comparison))
    report.extend([
        "",
        "### 房型",
        "",
    ])
    report.extend(render_table(flat_type_comparison))
    report.extend([
        "",
        "### 城镇",
        "",
    ])
    report.extend(render_table(town_comparison))
    report.extend([
        "",
        "### 面积与房龄",
        "",
    ])
    report.extend(render_table(numeric_comparison))
    report.extend([
        "",
        "测试集覆盖更晚的租赁年份，因此后续基线必须采用时间切分，而不应仅依赖随机切分。城镇、房型、面积和房龄均需在 EDA 中进一步检查其与租金的关系。",
    ])

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"已写入 {report_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
