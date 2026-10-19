"""Audit public HDB property snapshot without attaching it to model inputs."""
from pathlib import Path
import hashlib
import json
import pandas as pd
import sys

ROOT = Path(__file__).resolve().parents[2]
if __package__ in {None, ""}:
    sys.path.insert(0, str(ROOT))
from src.features.build_resale_history import canonical_street


def main():
    directory = ROOT / "data/raw/external/hdb_20261019"
    property_path = directory / "property.csv"
    data = pd.read_csv(property_path, dtype=str)
    for c in ["blk_no", "street"]:
        data[c] = data[c].str.strip().str.lower().str.replace(r"\s+", " ", regex=True)
    data["street"] = data.street.map(canonical_street)
    numeric = ["year_completed", "total_dwelling_units", "1room_rental", "2room_rental", "3room_rental", "other_room_rental"]
    for c in numeric:
        data[c] = pd.to_numeric(data[c], errors="raise")
    raw = pd.read_csv(ROOT / "data/interim/train_clean.csv", dtype={"BLOCK": str})
    raw["STREET"] = raw.STREET.map(canonical_street)
    keys = data[["blk_no", "street"]]
    if keys.duplicated().any():
        raise ValueError("Property snapshot has duplicate block/street keys")
    rental_counts = ["1room_rental", "2room_rental", "3room_rental", "other_room_rental"]
    data["rental_stock_total"] = data[rental_counts].sum(axis=1)
    matched = raw.join(data.set_index(["blk_no", "street"])[["total_dwelling_units", "rental_stock_total", "year_completed"]],
                       on=["BLOCK", "STREET"], validate="many_to_one")
    report = dict(source="https://data.gov.sg/datasets/d_17f5382f26140b1fdae0ba2ef6239d2f/view",
        sha256=hashlib.sha256(property_path.read_bytes()).hexdigest(), rows=len(data), columns=list(data.columns),
        duplicate_address_keys=int(keys.duplicated().sum()), missing_cells=int(data.isna().sum().sum()),
        training_address_coverage=float(matched.total_dwelling_units.notna().mean()),
        training_rows_in_blocks_with_rental_stock=float(matched.rental_stock_total.gt(0).mean()),
        year_completed_min=int(data.year_completed.min()), year_completed_max=int(data.year_completed.max()),
        receipt=json.loads((directory / "property_receipt.json").read_text()),
        used_in_training=False,
        history_ui_checked="2026-10-19; official See history lists ingestion timestamps only, not downloadable past files",
        earliest_visible_ingestion="2024-01-17",
        limitations=["Latest 2026 snapshot; historic values and release availability not verified",
                     "Stock counts are not rental-contract type labels and do not identify a unit",
                     "Matching coverage alone does not authorize point-in-time use"])
    output = ROOT / "reports/experiments/data_rethink_v3/property_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ["columns", "receipt"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
