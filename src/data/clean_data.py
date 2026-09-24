"""Create reproducible, non-destructive cleaned CSV files in data/interim."""

from __future__ import annotations

import argparse
import csv
import re
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MONTH_NUMBERS = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def normalise_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def normalise_flat_type(value: str) -> str:
    cleaned = normalise_text(value)
    match = re.fullmatch(r"([1-5])\s*-?\s*room", cleaned)
    return f"{match.group(1)}-room" if match else cleaned


def canonical_number(value: str, *, integer: bool = False) -> str:
    cleaned = value.strip().replace("$", "").replace(",", "")
    if not cleaned:
        return ""
    try:
        number = Decimal(cleaned)
    except InvalidOperation as error:
        raise ValueError(f"无法解析数值：{value!r}") from error
    if integer:
        return str(int(number))
    rendered = format(number.normalize(), "f")
    return rendered.rstrip("0").rstrip(".") if "." in rendered else rendered


def clean_main_rows(rows: list[dict[str, str]], *, has_target: bool) -> tuple[list[dict[str, str]], list[str]]:
    output: list[dict[str, str]] = []
    for row in rows:
        date = datetime.strptime(row["RENT_APPROVAL_DATE"].strip(), "%Y-%m")
        cleaned = {
            "RENT_APPROVAL_DATE": date.strftime("%Y-%m"),
            "TOWN": normalise_text(row["TOWN"]),
            "BLOCK": normalise_text(row["BLOCK"]),
            "STREET": normalise_text(row["STREET"]),
            "FLAT_TYPE": normalise_flat_type(row["FLAT_TYPE"]),
            "FLAT_MODEL": normalise_text(row["FLAT_MODEL"]),
            "FLOOR_AREA_SQM": canonical_number(row["FLOOR_AREA_SQM"]),
            "FURNISHED": normalise_text(row["FURNISHED"]),
            "LEASE_COMMENCE_DATE": canonical_number(row["LEASE_COMMENCE_DATE"], integer=True),
            "FEE": canonical_number(row["FEE"]),
            "RENT_YEAR": str(date.year),
            "RENT_MONTH": str(date.month),
            "RENT_MONTH_INDEX": str((date.year - 2021) * 12 + date.month),
        }
        flat_age = date.year - int(cleaned["LEASE_COMMENCE_DATE"])
        cleaned["FLAT_AGE"] = str(flat_age) if flat_age >= 0 else ""
        if has_target:
            cleaned["MONTHLY_RENT"] = canonical_number(row["MONTHLY_RENT"])
        output.append(cleaned)

    fields = [
        "RENT_APPROVAL_DATE",
        "TOWN",
        "BLOCK",
        "STREET",
        "FLAT_TYPE",
        "FLAT_MODEL",
        "FLOOR_AREA_SQM",
        "FURNISHED",
        "LEASE_COMMENCE_DATE",
        "FEE",
        "RENT_YEAR",
        "RENT_MONTH",
        "RENT_MONTH_INDEX",
        "FLAT_AGE",
    ]
    if has_target:
        fields.append("MONTHLY_RENT")
    return output, fields


def clean_hdb_rows(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[str]]:
    output = []
    for row in rows:
        town = normalise_text(row["TOWN"])
        block = normalise_text(row["BLOCK"])
        street = normalise_text(row["STREET"])
        completion_year = canonical_number(row["YEAR_COMPLETED"], integer=True)
        if completion_year == "-1":
            completion_year = ""
        output.append(
            {
                "TOWN": town,
                "BLOCK": block,
                "STREET": street,
                "POSTAL_CODE": row["POSTAL_CODE"].strip(),
                "LATITUDE": canonical_number(row["LATITUDE"]),
                "LONGITUDE": canonical_number(row["LONGITUDE"]),
                "MAX_FLOOR": canonical_number(row["MAX_FLOOR"], integer=True),
                "YEAR_COMPLETED": completion_year,
                "SUBZONE": normalise_text(row["SUBZONE"]),
                "PLANNING_AREA": normalise_text(row["PLANNING_AREA"]),
                "REGION": normalise_text(row["REGION"]),
                "ADDRESS_KEY": "|".join([town, block, street]),
            }
        )
    return output, [
        "TOWN",
        "BLOCK",
        "STREET",
        "POSTAL_CODE",
        "LATITUDE",
        "LONGITUDE",
        "MAX_FLOOR",
        "YEAR_COMPLETED",
        "SUBZONE",
        "PLANNING_AREA",
        "REGION",
        "ADDRESS_KEY",
    ]


def clean_mrt_rows(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[str]]:
    fields = ["CODE", "NAME", "STATUS", "LATITUDE", "LONGITUDE", "SUBZONE", "PLANNING_AREA", "REGION"]
    output = [
        {
            "CODE": row["CODE"].strip().lower(),
            "NAME": normalise_text(row["NAME"]),
            "STATUS": normalise_text(row["STATUS"]),
            "LATITUDE": canonical_number(row["LATITUDE"]),
            "LONGITUDE": canonical_number(row["LONGITUDE"]),
            "SUBZONE": normalise_text(row["SUBZONE"]),
            "PLANNING_AREA": normalise_text(row["PLANNING_AREA"]),
            "REGION": normalise_text(row["REGION"]),
        }
        for row in rows
    ]
    return output, fields


def clean_school_rows(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[str]]:
    fields = [
        "NAME",
        "URL",
        "STREET",
        "POSTAL_CODE",
        "MRT_STATIONS",
        "BUS_LINES",
        "PLANNING_AREA",
        "REGION",
        "TYPE_CODE",
        "NATRUE_CODE",
        "SESSION_CODE",
        "MAINLEVEL_CODE",
        "LATITUDE",
        "LONGITUDE",
    ]
    output = []
    for row in rows:
        cleaned = {field: normalise_text(row[field]) for field in fields if field not in {"URL", "POSTAL_CODE", "LATITUDE", "LONGITUDE"}}
        cleaned["URL"] = row["URL"].strip()
        cleaned["POSTAL_CODE"] = row["POSTAL_CODE"].strip()
        cleaned["LATITUDE"] = canonical_number(row["LATITUDE"])
        cleaned["LONGITUDE"] = canonical_number(row["LONGITUDE"])
        output.append(cleaned)
    return output, fields


def clean_mall_rows(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[str]]:
    fields = ["NAME", "LATITUDE", "LONGITUDE"]
    output = [
        {
            "NAME": normalise_text(row["NAME"]),
            "LATITUDE": canonical_number(row["LATITUDE"]),
            "LONGITUDE": canonical_number(row["LONGITUDE"]),
        }
        for row in rows
    ]
    return output, fields


def clean_coe_rows(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[str]]:
    fields = [
        "YEAR",
        "CATEGORY",
        "MONTH",
        "MONTH_NUM",
        "YEAR_MONTH",
        "ROUND",
        "QUOTA_PREMIUM",
        "PREVAILING_QUOTA_PREMIUM",
        "QUOTA",
        "BIDS_RECEIVED",
    ]
    output = []
    for row in rows:
        year = canonical_number(row["YEAR"], integer=True)
        month = normalise_text(row["MONTH"])
        month_num = MONTH_NUMBERS[month]
        output.append(
            {
                "YEAR": year,
                "CATEGORY": normalise_text(row["CATEGORY"]),
                "MONTH": month,
                "MONTH_NUM": str(month_num),
                "YEAR_MONTH": f"{year}-{month_num:02d}",
                "ROUND": canonical_number(row["ROUND"], integer=True),
                "QUOTA_PREMIUM": canonical_number(row["QUOTA_PREMIUM"]),
                "PREVAILING_QUOTA_PREMIUM": canonical_number(row["PREVAILING_QUOTA_PREMIUM"]),
                "QUOTA": canonical_number(row["QUOTA"], integer=True),
                "BIDS_RECEIVED": canonical_number(row["BIDS_RECEIVED"], integer=True),
            }
        )
    return output, fields


def clean_stock_rows(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[str]]:
    fields = ["NAME", "SYMBOL", "DATE", "OPEN", "HIGH", "LOW", "CLOSE", "ADJUSTED_CLOSE", "VOLUME"]
    output = []
    for row in rows:
        date = datetime.strptime(row["DATE"].strip(), "%Y-%m-%d").strftime("%Y-%m-%d")
        cleaned = {
            "NAME": normalise_text(row["NAME"]),
            "SYMBOL": row["SYMBOL"].strip().upper(),
            "DATE": date,
        }
        for field in ["OPEN", "HIGH", "LOW", "CLOSE", "ADJUSTED_CLOSE", "VOLUME"]:
            cleaned[field] = canonical_number(row[field])
        output.append(cleaned)
    return output, fields


def clean_file(source: Path, target: Path, cleaner, **kwargs: object) -> None:
    rows, _ = read_csv(source)
    cleaned_rows, fields = cleaner(rows, **kwargs)
    write_csv(target, cleaned_rows, fields)
    print(f"已写入 {target.relative_to(PROJECT_ROOT)}：{len(cleaned_rows):,} 行")


def main() -> None:
    parser = argparse.ArgumentParser(description="清洗项目 CSV 数据并写入 data/interim。")
    parser.add_argument("--input-dir", type=Path, default=PROJECT_ROOT / "data" / "raw")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "interim")
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    clean_file(input_dir / "train.csv", output_dir / "train_clean.csv", clean_main_rows, has_target=True)
    clean_file(input_dir / "test.csv", output_dir / "test_clean.csv", clean_main_rows, has_target=False)

    auxiliary_input = input_dir / "auxiliary"
    auxiliary_output = output_dir / "auxiliary"
    clean_file(auxiliary_input / "sg-hdb-block.csv", auxiliary_output / "sg-hdb-block_clean.csv", clean_hdb_rows)
    clean_file(auxiliary_input / "sg-mrt-stations.csv", auxiliary_output / "sg-mrt-stations_clean.csv", clean_mrt_rows)
    clean_file(auxiliary_input / "sg-schools.csv", auxiliary_output / "sg-schools_clean.csv", clean_school_rows)
    clean_file(auxiliary_input / "sg-shopping-malls.csv", auxiliary_output / "sg-shopping-malls_clean.csv", clean_mall_rows)
    clean_file(auxiliary_input / "sg-coe-prices.csv", auxiliary_output / "sg-coe-prices_clean.csv", clean_coe_rows)
    clean_file(auxiliary_input / "sg-stock-prices.csv", auxiliary_output / "sg-stock-prices_clean.csv", clean_stock_rows)


if __name__ == "__main__":
    main()
