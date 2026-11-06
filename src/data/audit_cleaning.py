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
    """Return counts and shares of a categorical field in the training and test sets."""
    train_counts = Counter(row[field] for row in train_rows)
    test_counts = Counter(row[field] for row in test_rows)
    rows = [[field, "Train count", "Train share", "Test count", "Test share", "Share difference (test - train)"]]
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
    """Return the minimum, median and maximum of numerical fields in the training and test sets."""
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
    parser = argparse.ArgumentParser(description="Generate the before/after cleaning data-quality report.")
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
        ("Training table", raw_dir / "train.csv", interim_dir / "train_clean.csv"),
        ("Test table", raw_dir / "test.csv", interim_dir / "test_clean.csv"),
        ("HDB blocks", raw_dir / "auxiliary" / "sg-hdb-block.csv", interim_dir / "auxiliary" / "sg-hdb-block_clean.csv"),
        ("MRT stations", raw_dir / "auxiliary" / "sg-mrt-stations.csv", interim_dir / "auxiliary" / "sg-mrt-stations_clean.csv"),
        ("Schools", raw_dir / "auxiliary" / "sg-schools.csv", interim_dir / "auxiliary" / "sg-schools_clean.csv"),
        ("Shopping malls", raw_dir / "auxiliary" / "sg-shopping-malls.csv", interim_dir / "auxiliary" / "sg-shopping-malls_clean.csv"),
        ("COE prices", raw_dir / "auxiliary" / "sg-coe-prices.csv", interim_dir / "auxiliary" / "sg-coe-prices_clean.csv"),
        ("Stock prices", raw_dir / "auxiliary" / "sg-stock-prices.csv", interim_dir / "auxiliary" / "sg-stock-prices_clean.csv"),
    ]

    report = [
        "# Data Quality After Cleaning",
        "",
        "This report compares the raw data in data/raw/ with the cleaned output in data/interim/. Cleaning removed no records and did not modify the raw files.",
        "",
    ]
    table = [["Dataset", "Raw rows", "Clean rows", "Raw missing cells", "Clean missing cells", "Raw duplicate rows", "Clean duplicate rows"]]
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
        ["Numerical field", "Train min", "Train median", "Train max", "Test min", "Test median", "Test max"],
        numeric_summary_row(clean_train, clean_test, "FLOOR_AREA_SQM"),
        numeric_summary_row(clean_train, clean_test, "FLAT_AGE"),
    ]

    report.extend([
        "",
        "## 1. Processing completed",
        "",
        "- Added RENT_YEAR, RENT_MONTH, RENT_MONTH_INDEX and FLAT_AGE to the main tables for baseline models and temporal validation.",
        f"- Standardised FLAT_TYPE: training categories reduced from {raw_flat_train} to {clean_flat_train}; test categories from {raw_flat_test} to {clean_flat_test}. Variants such as 3 room and 3-room were unified.",
        "- Town, block, street, flat model and furnishing fields were lowercased, trimmed and had repeated spaces collapsed.",
        "- Complete duplicate rows increased after cleaning because records that differed only in formatting became identical; no record was removed or merged.",
        f"- {missing_flat_age} training records have a rental year before the lease commencement year; their FLAT_AGE is left missing to avoid negative ages.",
        f"- {invalid_raw_years} HDB YEAR_COMPLETED values of -1 were converted to missing; the cleaned field has {blank_clean_years} missing values.",
        f"- {raw_currency_values} COE premium values with currency symbols or commas were converted to numbers; {coe_month_numbers:,}/{len(clean_coe):,} rows received MONTH_NUM and YEAR_MONTH.",
        "- MRT, school, mall and stock data had text, date and numeric formats standardised, without aggregation or deletion.",
        "",
        "## 2. Consistency and usability checks",
        "",
        f"- Cleaned training address matches: {train_match:,}/{train_total:,} ({train_match / train_total:.1%}).",
        f"- Cleaned test address matches: {test_match:,}/{test_total:,} ({test_match / test_total:.1%}).",
        f"- Cleaned MRT status: {mrt_status['open']} open, {mrt_status['planned']} planned. The first location features should use open stations only.",
        f"- The stock table still has {stock_missing} rows with missing closing prices; these are missing in the raw data and are handled when aggregating by security and month.",
        "",
        "## 3. Remaining limitations and next steps",
        "",
        "- FEE is always 0 and FURNISHED is always yes; both are kept for lineage but should be excluded from the first model.",
        "- Duplicates are not removed. Identical inputs can have different rents, so deduplicating on features would discard valid observations.",
        "- COE can have several rounds for the same month and category; feature engineering must choose the last round, the mean or another aggregation.",
        "- Stock data are only cleaned here; monthly returns and volatility are built in stage 5 if validation supports them.",
        "- Planned MRT stations may leak future information; they should not be used for early rental records without reliable opening dates.",
        "- Next step: build a time-split baseline on data/interim/train_clean.csv and data/interim/test_clean.csv using main-table features only.",
        "",
        "## 4. Input and output paths",
        "",
        "- Input: data/raw/, read-only Kaggle files.",
        "- This stage: data/interim/, cleaned main and auxiliary tables.",
        "- Later feature engineering: data/processed/, merged modelling tables.",
        "- This report: reports/data_quality_after_cleaning.md.",
    ])

    report.extend([
        "",
        "## 5. Train/test distribution comparison",
        "",
        "These tables identify structural differences between training and test data. This is an audit only and does not analyse rent; rent distributions and trends are covered by the stage 2 EDA.",
        "",
        "### Rental year",
        "",
    ])
    report.extend(render_table(year_comparison))
    report.extend([
        "",
        "### Flat type",
        "",
    ])
    report.extend(render_table(flat_type_comparison))
    report.extend([
        "",
        "### Town",
        "",
    ])
    report.extend(render_table(town_comparison))
    report.extend([
        "",
        "### Floor area and flat age",
        "",
    ])
    report.extend(render_table(numeric_comparison))
    report.extend([
        "",
        "The test set covers later rental years, so baselines must use a temporal split rather than only a random split. Town, flat type, floor area and age are examined against rent in the EDA.",
    ])

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Wrote {report_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
