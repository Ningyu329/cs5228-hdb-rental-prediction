"""Join HDB addresses and build reproducible location features without rent labels.

Run after src/data/clean_data.py. Snapshot MRT features exclude planned stations,
but need historical opening dates before they can represent past accessibility.
All distances are great-circle distances in kilometres, not walking distances.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EARTH_RADIUS_KM = 6371.0088
ADDRESS_COLUMNS = ["TOWN", "BLOCK", "STREET"]
HDB_COLUMNS = [
    "LATITUDE", "LONGITUDE", "SUBZONE", "PLANNING_AREA", "REGION",
    "YEAR_COMPLETED", "MAX_FLOOR", "POSTAL_CODE",
]
INPUT_FILES = {
    "train": "train_clean.csv",
    "test": "test_clean.csv",
    "hdb": "auxiliary/sg-hdb-block_clean.csv",
    "mrt": "auxiliary/sg-mrt-stations_clean.csv",
    "school": "auxiliary/sg-schools_clean.csv",
    "mall": "auxiliary/sg-shopping-malls_clean.csv",
}


def read_table(path: Path) -> pd.DataFrame:
    # Read identifiers as text: block 001 and postal codes must not lose zeros.
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def require_columns(df: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise ValueError(f"{name}: missing required columns: {', '.join(missing)}")


def normalise_text(values: pd.Series) -> pd.Series:
    return values.astype("string").str.strip().str.lower().str.replace(r"\s+", " ", regex=True)


def address_key(df: pd.DataFrame) -> pd.Series:
    require_columns(df, ADDRESS_COLUMNS, "address table")
    parts = [normalise_text(df[column]).replace("", pd.NA) for column in ADDRESS_COLUMNS]
    # Missing address parts stay missing. Never join an incomplete key to HDB.
    return parts[0] + "|" + parts[1] + "|" + parts[2]


def valid_coordinates(df: pd.DataFrame) -> pd.Series:
    return df["LATITUDE"].between(-90, 90) & df["LONGITUDE"].between(-180, 180)


def numeric_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for column in ["LATITUDE", "LONGITUDE"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df.loc[~valid_coordinates(df), ["LATITUDE", "LONGITUDE"]] = np.nan
    return df


def prepare_hdb(df: pd.DataFrame) -> pd.DataFrame:
    require_columns(df, ADDRESS_COLUMNS + HDB_COLUMNS[:6], "HDB")
    df = df.copy()
    df["ADDRESS_KEY"] = address_key(df)
    if df["ADDRESS_KEY"].isna().any():
        raise ValueError("HDB: incomplete address keys; repair the auxiliary table first")
    if df["ADDRESS_KEY"].duplicated().any():
        raise ValueError("HDB: duplicate address keys would expand rental rows; repair them first")
    df = numeric_coordinates(df)
    for column in ["SUBZONE", "PLANNING_AREA", "REGION"]:
        df[column] = normalise_text(df[column]).replace("", pd.NA)
    years = pd.to_numeric(df["YEAR_COMPLETED"], errors="coerce")
    df["YEAR_COMPLETED"] = years.where(years.between(1800, 9999) & years.eq(years.round()))
    df["YEAR_COMPLETED_MISSING"] = df["YEAR_COMPLETED"].isna().astype(int)
    if "MAX_FLOOR" not in df:
        df["MAX_FLOOR"] = np.nan
    floors = pd.to_numeric(df["MAX_FLOOR"], errors="coerce")
    df["MAX_FLOOR"] = floors.where(floors.gt(0))
    if "POSTAL_CODE" not in df:
        df["POSTAL_CODE"] = pd.NA
    return df[["ADDRESS_KEY", *HDB_COLUMNS, "YEAR_COMPLETED_MISSING"]].reset_index(drop=True)


def prepare_sites(
    df: pd.DataFrame,
    kind: str,
    mrt_mode: str,
    opening_dates: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict]:
    required = ["LATITUDE", "LONGITUDE"] + (["CODE", "STATUS"] if kind == "mrt" else ["NAME"])
    require_columns(df, required, kind)
    df = df.copy()
    audit = {"input_rows": len(df)}
    if kind == "mrt":
        df["CODE"] = normalise_text(df["CODE"])
        df["STATUS"] = normalise_text(df["STATUS"])
        audit["excluded_non_open_rows"] = int(df["STATUS"].ne("open").sum())
        df = df.loc[df["STATUS"].eq("open")].copy()
        if opening_dates is not None:
            dates = opening_dates.copy()
            require_columns(dates, ["CODE", "OPENING_DATE"], "MRT opening dates")
            dates["CODE"] = normalise_text(dates["CODE"])
            if dates["CODE"].fillna("").eq("").any() or dates["CODE"].duplicated().any():
                raise ValueError("MRT opening dates: CODE must be non-empty and unique")
            df = df.drop(columns="OPENING_DATE", errors="ignore").merge(
                dates[["CODE", "OPENING_DATE"]], on="CODE", how="left", validate="many_to_one", sort=False,
            )
        if mrt_mode == "historical":
            require_columns(df, ["OPENING_DATE"], "historical MRT mode")
            dates = pd.to_datetime(df["OPENING_DATE"], format="ISO8601", errors="coerce")
            if dates.isna().any():
                raise ValueError("Historical MRT mode requires a valid OPENING_DATE for every open station")
            df["OPENING_DATE"] = dates
        audit["open_rows"] = len(df)
    else:
        df["NAME"] = normalise_text(df["NAME"])
        if df["NAME"].fillna("").eq("").any():
            raise ValueError(f"{kind}: facility NAME must be non-empty")
    df = numeric_coordinates(df)
    audit["invalid_coordinate_rows"] = int((~valid_coordinates(df)).sum())
    df = df.loc[valid_coordinates(df)].copy()
    if kind == "mrt":
        # Multiple line codes at the same coordinates count as one location.
        # In historical mode the first opening date determines availability.
        if mrt_mode == "historical":
            df = df.sort_values("OPENING_DATE")
        subset = ["LATITUDE", "LONGITUDE"]
    else:
        # Distinct schools sharing coordinates remain distinct facilities.
        subset = ["NAME", "LATITUDE", "LONGITUDE"]
    before = len(df)
    df = df.drop_duplicates(subset=subset).reset_index(drop=True)
    audit["duplicate_location_rows_removed"] = before - len(df)
    audit["usable_locations"] = len(df)
    return df, audit


def proximity_features(
    origins: pd.DataFrame,
    sites: pd.DataFrame,
    prefix: str,
    radii_m: tuple[int, ...],
    chunk_size: int = 1024,
) -> pd.DataFrame:
    """Haversine nearest distance and inclusive radius counts in bounded memory."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    columns = [f"NEAREST_{prefix}_DISTANCE_KM", *[f"{prefix}_COUNT_WITHIN_{r}M" for r in radii_m]]
    output = pd.DataFrame(np.nan, index=origins.index, columns=columns)
    positions = np.flatnonzero(valid_coordinates(origins).to_numpy())
    if len(positions) == 0:
        return output
    if sites.empty:
        # With no eligible stations in an early month, the known count is zero
        # and nearest distance is undefined. Missing origin coordinates stay NaN.
        output.iloc[positions, 1:] = 0
        return output
    candidates = np.radians(sites[["LATITUDE", "LONGITUDE"]].to_numpy(dtype=float))
    for start in range(0, len(positions), chunk_size):
        batch = positions[start:start + chunk_size]
        points = np.radians(origins.iloc[batch][["LATITUDE", "LONGITUDE"]].to_numpy(dtype=float))
        dlat = candidates[None, :, 0] - points[:, None, 0]
        dlon = candidates[None, :, 1] - points[:, None, 1]
        a = np.sin(dlat / 2) ** 2 + (
            np.cos(points[:, None, 0]) * np.cos(candidates[None, :, 0]) * np.sin(dlon / 2) ** 2
        )
        distances = 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
        output.iloc[batch, 0] = distances.min(axis=1)
        for column, radius in enumerate(radii_m, start=1):
            output.iloc[batch, column] = (distances <= radius / 1000 + 1e-10).sum(axis=1)
    return output


def attach_locations(
    main: pd.DataFrame,
    blocks: pd.DataFrame,
    mrt: pd.DataFrame,
    mrt_mode: str,
    radii_m: tuple[int, ...],
) -> pd.DataFrame:
    require_columns(main, ADDRESS_COLUMNS + ["RENT_APPROVAL_DATE"], "rental table")
    reserved = set(blocks.columns) - {"ADDRESS_KEY"}
    reserved.update({"ADDRESS_MATCHED", "LOCATION_COORDINATES_VALID", "BUILDING_AGE", "BUILDING_AGE_MISSING", "_hdb_merge"})
    reserved.update(["NEAREST_OPEN_MRT_DISTANCE_KM", *[f"OPEN_MRT_COUNT_WITHIN_{r}M" for r in radii_m]])
    overlap = sorted(set(main.columns) & reserved)
    if overlap:
        raise ValueError(f"Rental input already contains location output columns: {', '.join(overlap)}")
    out = main.copy()
    out["ADDRESS_KEY"] = address_key(out)
    out = out.merge(blocks, on="ADDRESS_KEY", how="left", sort=False, validate="many_to_one", indicator="_hdb_merge")
    if len(out) != len(main):
        raise AssertionError("Address join changed the number of rental records")
    out["ADDRESS_MATCHED"] = out.pop("_hdb_merge").eq("both").astype(int)
    out["LOCATION_COORDINATES_VALID"] = valid_coordinates(out).astype(int)
    out["YEAR_COMPLETED_MISSING"] = out["YEAR_COMPLETED"].isna().astype(int)
    approval = pd.to_datetime(out["RENT_APPROVAL_DATE"], format="ISO8601", errors="coerce")
    if approval.isna().any():
        raise ValueError("Rental table contains missing or invalid RENT_APPROVAL_DATE values")
    age = approval.dt.year - out["YEAR_COMPLETED"]
    out["BUILDING_AGE"] = age.where(age.ge(0))
    out["BUILDING_AGE_MISSING"] = out["BUILDING_AGE"].isna().astype(int)
    if mrt_mode == "historical":
        # Initialise even for an empty split so train/test output schemas agree.
        out = pd.concat([out, proximity_features(out, mrt.iloc[:0], "OPEN_MRT", radii_m)], axis=1)
        # Approval data has month precision. Only stations already open at the
        # start of that month qualify; a mid-month opening enters next month.
        month_start = approval.dt.to_period("M").dt.to_timestamp()
        for month, indices in out.groupby(month_start, sort=True).groups.items():
            eligible = mrt.loc[mrt["OPENING_DATE"].le(month)]
            unique = out.loc[indices, ["LATITUDE", "LONGITUDE"]].drop_duplicates().reset_index(drop=True)
            features = proximity_features(unique, eligible, "OPEN_MRT", radii_m)
            lookup = pd.concat([unique, features], axis=1)
            values = out.loc[indices, ["LATITUDE", "LONGITUDE"]].merge(
                lookup, on=["LATITUDE", "LONGITUDE"], how="left", validate="many_to_one", sort=False,
            )
            for column in features:
                out.loc[indices, column] = values[column].to_numpy()
    return out


def coverage_summary(df: pd.DataFrame) -> dict:
    matched = int(df["ADDRESS_MATCHED"].sum())
    unique = df[["ADDRESS_KEY", "ADDRESS_MATCHED"]].drop_duplicates()
    valid = int(df["LOCATION_COORDINATES_VALID"].sum())
    town_table = df.assign(_town=normalise_text(df["TOWN"]).fillna("<missing>").replace("", "<missing>"))
    by_town = []
    for town, group in town_table.groupby("_town", sort=True):
        by_town.append({"town": town, "rows": len(group), "matched_rows": int(group["ADDRESS_MATCHED"].sum())})
    return {
        "rows": len(df), "matched_rows": matched,
        "match_rate": matched / len(df) if len(df) else None,
        "unique_address_keys_including_missing": len(unique),
        "matched_unique_addresses": int(unique["ADDRESS_MATCHED"].sum()),
        "valid_coordinate_rows": valid,
        "valid_coordinate_rate": valid / len(df) if len(df) else None,
        "missing_building_age_rows": int(df["BUILDING_AGE_MISSING"].sum()),
        "by_town": by_town,
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def render_report(manifest: dict) -> str:
    lines = [
        "# Address Matching and Location Features Run Report", "",
        "This report is generated from the inputs of this run only; historical audit match rates are not substituted.", "",
        "| Dataset | Rows | Matched rows | Address match rate | Valid coordinate rows | Missing building age rows |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split, stats in manifest["coverage"].items():
        rate = f"{stats['match_rate']:.2%}" if stats["match_rate"] is not None else "N/A"
        lines.append(f"| {split} | {stats['rows']:,} | {stats['matched_rows']:,} | {rate} | {stats['valid_coordinate_rows']:,} | {stats['missing_building_age_rows']:,} |")
    lines.extend(["", "## Features and processing rules", ""])
    lines.extend(f"- {rule}" for rule in manifest["rules"])
    lines.extend(["", "## Temporal availability and limitations", ""])
    lines.extend(f"- {note}" for note in manifest["limitations"])
    lines.extend(["", "## Auxiliary data checks", "", "| Source | Input rows | Valid unique locations | Invalid coordinate rows |", "| --- | ---: | ---: | ---: |"])
    for name, stats in manifest["sites"].items():
        lines.append(f"| {name} | {stats['input_rows']} | {stats['usable_locations']} | {stats['invalid_coordinate_rows']} |")
    lines.extend(["", "## Address matching by town", "", "| Dataset | Town | Rows | Matched rows |", "| --- | --- | ---: | ---: |"])
    for split, stats in manifest["coverage"].items():
        for town in stats["by_town"]:
            label = town["town"].replace("|", "/")
            lines.append(f"| {split} | {label} | {town['rows']} | {town['matched_rows']} |")
    lines.extend(["", "## Next validation step", "", "Compare validation RMSE of main-table features and these location features on the same temporal split. This script only builds features; it trains no model and makes no RMSE claim.", ""])
    return "\n".join(lines)


def run_pipeline(
    interim_dir: Path,
    output_dir: Path,
    report_dir: Path,
    radii_m: tuple[int, ...] = (500, 1000),
    mrt_mode: str = "snapshot",
    mrt_opening_dates: Path | None = None,
) -> dict:
    if mrt_mode not in {"snapshot", "historical"}:
        raise ValueError("mrt_mode must be snapshot or historical")
    if not radii_m or any(not isinstance(r, int) or r <= 0 for r in radii_m) or len(set(radii_m)) != len(radii_m):
        raise ValueError("radii_m must contain distinct positive integer radii in metres")
    radii_m = tuple(sorted(radii_m))
    interim_dir, output_dir, report_dir = (p.resolve() for p in [interim_dir, output_dir, report_dir])
    paths = {name: interim_dir / filename for name, filename in INPUT_FILES.items()}
    if mrt_opening_dates is not None:
        paths["mrt_opening_dates"] = mrt_opening_dates.resolve()
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing input CSV files. Run src/data/clean_data.py first:\n" + "\n".join(missing))
    raw_dir = (PROJECT_ROOT / "data" / "raw").resolve()
    for directory in [output_dir, report_dir]:
        if directory == interim_dir or directory.is_relative_to(interim_dir) or directory.is_relative_to(raw_dir):
            raise ValueError("Outputs must be outside the input directory and data/raw")
    outputs = [output_dir / f"{split}_location_features.csv" for split in ["train", "test"]]
    outputs += [output_dir / f"{split}_unmatched_addresses.csv" for split in ["train", "test"]]
    outputs += [report_dir / "location_features_manifest.json", report_dir / "location_features_report.md"]
    if set(outputs) & set(paths.values()):
        raise ValueError("An output would overwrite an input file")

    tables = {name: read_table(path) for name, path in paths.items()}
    hdb = prepare_hdb(tables["hdb"])
    sites, site_audit = {}, {}
    for kind in ["mrt", "school", "mall"]:
        sites[kind], site_audit[kind] = prepare_sites(tables[kind], kind, mrt_mode, tables.get("mrt_opening_dates"))
    blocks = hdb.copy()
    for kind, prefix in [("mrt", "OPEN_MRT"), ("school", "SCHOOL"), ("mall", "MALL")]:
        if kind == "mrt" and mrt_mode == "historical":
            continue
        blocks = pd.concat([blocks, proximity_features(hdb, sites[kind], prefix, radii_m)], axis=1)
    featured = {split: attach_locations(tables[split], blocks, sites["mrt"], mrt_mode, radii_m) for split in ["train", "test"]}
    # Left joins must keep duplicate rentals, labels, IDs and original row order.
    for split, out in featured.items():
        original_columns = [c for c in tables[split].columns if c != "ADDRESS_KEY"]
        pd.testing.assert_frame_equal(out[original_columns].reset_index(drop=True), tables[split][original_columns].reset_index(drop=True))
    features = [c for c in featured["train"].columns if c not in tables["train"].columns and c != "ADDRESS_KEY"]
    time_note = (
        "MRT stations are filtered by OPENING_DATE <= first day of the rental month and non-open stations are excluded; stations opening mid-month count from the next month."
        if mrt_mode == "historical" else
        "MRT snapshot mode only keeps stations with current STATUS=open, which does not prove they were open in historical rental months, so temporal leakage remains possible. Formal temporal validation should supply opening dates and use historical mode."
    )
    manifest = {
        "feature_version": "location_v1", "mrt_mode": mrt_mode,
        "radii_m": list(radii_m), "earth_radius_km": EARTH_RADIUS_KM,
        "source_sha256": {name: file_sha256(path) for name, path in paths.items()},
        "feature_columns": features,
        "hdb": {"rows": len(hdb), "invalid_coordinate_rows": int((~valid_coordinates(hdb)).sum()), "missing_completion_years": int(hdb["YEAR_COMPLETED"].isna().sum())},
        "sites": site_audit, "coverage": {split: coverage_summary(df) for split, df in featured.items()},
        "rules": [
            "Addresses are matched exactly on standardised TOWN + BLOCK + STREET; missing addresses are not matched and no fuzzy guessing is used.",
            "HDB address keys must be unique. Main-table row count, order, duplicates and labels are preserved; rent is never used to build location features.",
            "Coordinates, planning area, region, completion year, maximum floor and postal code are kept. ADDRESS_MATCHED and LOCATION_COORDINATES_VALID are flagged separately.",
            "BUILDING_AGE = rental approval year - YEAR_COMPLETED, which differs from FLAT_AGE based on lease commencement. Invalid, missing or negative building ages are left empty with a missing indicator.",
            "Distances use the Haversine great-circle distance in km. Facility counts within a radius (inclusive) are density proxies, not walking distance or service quality.",
            "MRT stations at the same coordinates count once, so interchanges with several line codes are not double counted; schools and malls are deduplicated by name and coordinates.",
            "Distances and counts are empty for invalid or unmatched coordinates. With no usable facility the nearest distance is empty and counts are 0 for valid coordinates.",
        ],
        "limitations": [time_note, "School, mall and HDB area attributes come from static auxiliary tables without full historical versions. The location features cannot be claimed to remove temporal leakage entirely.", "Feature generation fits no imputer, encoder or clustering. Later modelling must fit such transforms on the training fold only."],
    }
    # Validate every source and both joins before producing any outputs.
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    for split, out in featured.items():
        out.to_csv(output_dir / f"{split}_location_features.csv", index=False)
        unmatched = out.loc[out["ADDRESS_MATCHED"].eq(0), ADDRESS_COLUMNS + ["ADDRESS_KEY"]].drop_duplicates()
        unmatched.to_csv(output_dir / f"{split}_unmatched_addresses.csv", index=False)
    (report_dir / "location_features_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (report_dir / "location_features_report.md").write_text(render_report(manifest), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interim-dir", type=Path, default=PROJECT_ROOT / "data/interim")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data/processed")
    parser.add_argument("--report-dir", type=Path, default=PROJECT_ROOT / "reports/location_features")
    parser.add_argument("--radii-m", nargs="+", type=int, default=[500, 1000])
    parser.add_argument("--mrt-mode", choices=["snapshot", "historical"], default="snapshot")
    parser.add_argument("--mrt-opening-dates", type=Path, help="Optional CSV with CODE,OPENING_DATE (ISO dates)")
    args = parser.parse_args()
    try:
        result = run_pipeline(args.interim_dir, args.output_dir, args.report_dir, tuple(args.radii_m), args.mrt_mode, args.mrt_opening_dates)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    for split, stats in result["coverage"].items():
        rate = f"{stats['match_rate']:.2%}" if stats["match_rate"] is not None else "N/A"
        print(f"{split}: {stats['matched_rows']:,}/{stats['rows']:,} addresses matched ({rate})")
    print(f"MRT mode: {args.mrt_mode}. See report for temporal limitations.")
    print(f"Feature tables: {args.output_dir}")
    print(f"Coverage report: {args.report_dir / 'location_features_report.md'}")


if __name__ == "__main__":
    main()
