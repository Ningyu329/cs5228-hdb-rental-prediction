"""Regression tests for location joins, distances, date policy and CLI outputs.

Fixtures are deliberately synthetic and never stand in for competition results.
Run: python -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.build_location_features import (
    EARTH_RADIUS_KM, PROJECT_ROOT, address_key, attach_locations, file_sha256,
    prepare_hdb, prepare_sites, proximity_features, read_table, run_pipeline,
)


def hdb_table() -> pd.DataFrame:
    return pd.DataFrame({
        "TOWN": ["Town A", "Town B"], "BLOCK": ["001", "2"],
        "STREET": ["First Street", "Second Street"],
        "LATITUDE": ["1.30", "999"], "LONGITUDE": ["103.80", "103.90"],
        "SUBZONE": [" Zone A ", "zone b"], "PLANNING_AREA": ["Area A", "Area B"],
        "REGION": ["Central Region", "north region"], "YEAR_COMPLETED": ["2000", "-1"],
        "MAX_FLOOR": ["12", "16"], "POSTAL_CODE": ["012345", "654321"],
    })


def rental_table() -> pd.DataFrame:
    return pd.DataFrame({
        "Id": ["5", "2", "9", "7", "8"],
        "TOWN": [" TOWN   A ", "town a", "town b", "absent", "town a"],
        "BLOCK": ["001", "001", "2", "3", "001"],
        "STREET": ["First   Street", "first street", "second street", "third street", "first street"],
        "RENT_APPROVAL_DATE": ["2025-01", "2025-02", "2025-01", "2025-03", "1999-01"],
        "MONTHLY_RENT": ["3000", "3100", "2500", "2700", "2000"],
    })


def mrt_table() -> pd.DataFrame:
    return pd.DataFrame({
        "CODE": ["A1", "A2", "A3", "P1"], "STATUS": [" Open ", "open", "open", "planned"],
        "LATITUDE": ["1.301", "1.301", "1.303", "1.300"],
        "LONGITUDE": ["103.80"] * 4,
    })


def facilities() -> pd.DataFrame:
    return pd.DataFrame({"NAME": ["School A", "school a", "School B"], "LATITUDE": ["1.30"] * 3, "LONGITUDE": ["103.80"] * 3})


def write_fixture(directory: Path) -> None:
    aux = directory / "auxiliary"
    aux.mkdir(parents=True)
    rental_table().to_csv(directory / "train_clean.csv", index=False)
    rental_table().drop(columns="MONTHLY_RENT").to_csv(directory / "test_clean.csv", index=False)
    hdb_table().to_csv(aux / "sg-hdb-block_clean.csv", index=False)
    mrt_table().to_csv(aux / "sg-mrt-stations_clean.csv", index=False)
    facilities().to_csv(aux / "sg-schools_clean.csv", index=False)
    facilities().to_csv(aux / "sg-shopping-malls_clean.csv", index=False)


class LocationFeaturesTests(unittest.TestCase):
    def test_normalisation_and_incomplete_keys(self):
        table = pd.DataFrame({"TOWN": ["  ToWn   A", "", None], "BLOCK": ["001", "2", "3"], "STREET": ["  First  Street ", "road", "road"]})
        keys = address_key(table)
        self.assertEqual(keys.iloc[0], "town a|001|first street")
        self.assertTrue(keys.iloc[1:].isna().all())

    def test_hdb_rejects_duplicate_normalised_addresses(self):
        table = pd.concat([hdb_table(), hdb_table().iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "duplicate address"):
            prepare_hdb(table)

    def test_hdb_missing_years_and_invalid_coordinates(self):
        hdb = prepare_hdb(hdb_table())
        self.assertTrue(pd.isna(hdb.loc[1, "YEAR_COMPLETED"]))
        self.assertEqual(hdb.loc[1, "YEAR_COMPLETED_MISSING"], 1)
        self.assertTrue(hdb.loc[1, ["LATITUDE", "LONGITUDE"]].isna().all())
        self.assertEqual(hdb.loc[0, "POSTAL_CODE"], "012345")

    def test_join_preserves_order_duplicates_labels_and_unknown_addresses(self):
        source = rental_table()
        source = pd.concat([source, source.iloc[[0]]], ignore_index=True)
        hdb = prepare_hdb(hdb_table())
        sites, _ = prepare_sites(mrt_table(), "mrt", "snapshot")
        out = attach_locations(source, hdb, sites, "snapshot", (500, 1000))
        pd.testing.assert_frame_equal(out[source.columns], source)
        self.assertEqual(out["ADDRESS_MATCHED"].tolist(), [1, 1, 1, 0, 1, 1])
        self.assertEqual(out["LOCATION_COORDINATES_VALID"].tolist(), [1, 1, 0, 0, 1, 1])
        self.assertEqual(out.loc[0, "BUILDING_AGE"], 25)
        self.assertTrue(pd.isna(out.loc[4, "BUILDING_AGE"]))

    def test_haversine_known_distance_chunking_and_inclusive_radius(self):
        origins = pd.DataFrame({"LATITUDE": [0.0, 1.0, np.nan], "LONGITUDE": [0.0, 0.0, 0.0]})
        sites = pd.DataFrame({"LATITUDE": [0.0], "LONGITUDE": [0.0]})
        result = proximity_features(origins, sites, "SCHOOL", (500, 1000), chunk_size=1)
        self.assertEqual(result.loc[0, "NEAREST_SCHOOL_DISTANCE_KM"], 0)
        self.assertAlmostEqual(result.loc[1, "NEAREST_SCHOOL_DISTANCE_KM"], EARTH_RADIUS_KM * np.pi / 180, places=6)
        self.assertTrue(result.loc[2].isna().all())
        boundary = pd.DataFrame({"LATITUDE": [0.0], "LONGITUDE": [np.degrees(0.5 / EARTH_RADIUS_KM)]})
        counts = proximity_features(origins.iloc[[0]], boundary, "SCHOOL", (500, 1000))
        self.assertEqual(counts.iloc[0]["SCHOOL_COUNT_WITHIN_500M"], 1)

    def test_mrt_excludes_planned_and_deduplicates_interchanges(self):
        stations, audit = prepare_sites(mrt_table(), "mrt", "snapshot")
        self.assertEqual(len(stations), 2)
        self.assertEqual(audit["excluded_non_open_rows"], 1)
        self.assertEqual(audit["duplicate_location_rows_removed"], 1)
        result = proximity_features(prepare_hdb(hdb_table()), stations, "OPEN_MRT", (500, 1000))
        self.assertGreater(result.loc[0, "NEAREST_OPEN_MRT_DISTANCE_KM"], 0)
        self.assertEqual(result.loc[0, "OPEN_MRT_COUNT_WITHIN_500M"], 2)

    def test_named_facilities_sharing_coordinates_remain_distinct(self):
        schools, audit = prepare_sites(facilities(), "school", "snapshot")
        self.assertEqual(len(schools), 2)
        self.assertEqual(audit["duplicate_location_rows_removed"], 1)

    def test_historical_mode_requires_complete_valid_dates(self):
        with self.assertRaisesRegex(ValueError, "OPENING_DATE"):
            prepare_sites(mrt_table(), "mrt", "historical")
        stations = mrt_table().assign(OPENING_DATE=["bad", "2020-01-01", "2025-01-15", "2030-01-01"])
        with self.assertRaisesRegex(ValueError, "valid OPENING_DATE"):
            prepare_sites(stations, "mrt", "historical")

    def test_historical_station_availability_and_month_start_policy(self):
        stations = mrt_table().assign(OPENING_DATE=["2024-01-01", "2026-01-01", "2025-01-15", "2024-01-01"])
        mrt, _ = prepare_sites(stations, "mrt", "historical")
        out = attach_locations(rental_table(), prepare_hdb(hdb_table()), mrt, "historical", (500, 1000))
        self.assertEqual(out.loc[0, "OPEN_MRT_COUNT_WITHIN_500M"], 1)
        self.assertEqual(out.loc[1, "OPEN_MRT_COUNT_WITHIN_500M"], 2)
        self.assertEqual(out.loc[4, "OPEN_MRT_COUNT_WITHIN_500M"], 0)
        self.assertTrue(pd.isna(out.loc[4, "NEAREST_OPEN_MRT_DISTANCE_KM"]))
        self.assertTrue(pd.isna(out.loc[2, "OPEN_MRT_COUNT_WITHIN_500M"]))

    def test_opening_date_sidecar_matches_codes(self):
        dates = pd.DataFrame({"CODE": [" a1 ", "a2", "a3"], "OPENING_DATE": ["2020-01-01", "2021-01-01", "2024-01-01"]})
        stations, _ = prepare_sites(mrt_table(), "mrt", "historical", dates)
        self.assertEqual(len(stations), 2)
        self.assertEqual(stations.iloc[0]["OPENING_DATE"], pd.Timestamp("2020-01-01"))

    def test_empty_historical_split_keeps_feature_schema(self):
        stations = mrt_table().assign(OPENING_DATE="2020-01-01")
        mrt, _ = prepare_sites(stations, "mrt", "historical")
        out = attach_locations(rental_table().iloc[:0], prepare_hdb(hdb_table()), mrt, "historical", (500, 1000))
        self.assertEqual(len(out), 0)
        self.assertIn("NEAREST_OPEN_MRT_DISTANCE_KM", out.columns)
        self.assertIn("OPEN_MRT_COUNT_WITHIN_1000M", out.columns)

    def test_empty_site_pool_is_not_a_zero_distance(self):
        origins = prepare_hdb(hdb_table())
        sites = pd.DataFrame(columns=["LATITUDE", "LONGITUDE"])
        result = proximity_features(origins, sites, "MALL", (500,))
        self.assertTrue(pd.isna(result.loc[0, "NEAREST_MALL_DISTANCE_KM"]))
        self.assertEqual(result.loc[0, "MALL_COUNT_WITHIN_500M"], 0)
        self.assertTrue(result.loc[1].isna().all())

    def test_input_location_columns_cannot_be_silently_overwritten(self):
        source = rental_table().assign(LATITUDE="1.0")
        with self.assertRaisesRegex(ValueError, "already contains"):
            attach_locations(source, prepare_hdb(hdb_table()), pd.DataFrame(), "snapshot", (500,))

    def test_full_pipeline_outputs_coverage_hashes_and_target_independence(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            interim, output, report = root / "interim", root / "processed", root / "reports"
            write_fixture(interim)
            before = {path: file_sha256(path) for path in interim.rglob("*.csv")}
            result = run_pipeline(interim, output, report)
            self.assertEqual(result["coverage"]["train"]["matched_rows"], 4)
            self.assertEqual(result["coverage"]["train"]["match_rate"], 0.8)
            self.assertEqual(result["sites"]["mrt"]["usable_locations"], 2)
            manifest = json.loads((report / "location_features_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["mrt_mode"], "snapshot")
            self.assertIn("temporal leakage", manifest["limitations"][0])
            self.assertIn("NEAREST_OPEN_MRT_DISTANCE_KM", manifest["feature_columns"])
            train = read_table(output / "train_location_features.csv")
            test = read_table(output / "test_location_features.csv")
            pd.testing.assert_frame_equal(train.drop(columns="MONTHLY_RENT"), test)
            self.assertEqual(test.loc[0, "BLOCK"], "001")
            self.assertEqual(test.loc[0, "POSTAL_CODE"], "012345")
            self.assertEqual(len(read_table(output / "train_unmatched_addresses.csv")), 1)
            self.assertEqual(before, {path: file_sha256(path) for path in interim.rglob("*.csv")})
            changed = read_table(interim / "train_clean.csv")
            changed["MONTHLY_RENT"] = "999999"
            changed.to_csv(interim / "train_clean.csv", index=False)
            run_pipeline(interim, root / "again", root / "again-reports")
            again = read_table(root / "again/train_location_features.csv")
            pd.testing.assert_frame_equal(train.drop(columns="MONTHLY_RENT"), again.drop(columns="MONTHLY_RENT"))

    def test_cli_historical_mode(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            interim = root / "interim"
            write_fixture(interim)
            dates = root / "opening_dates.csv"
            pd.DataFrame({"CODE": ["A1", "A2", "A3"], "OPENING_DATE": ["2024-01-01", "2024-01-01", "2025-01-15"]}).to_csv(dates, index=False)
            result = subprocess.run([
                sys.executable, str(PROJECT_ROOT / "src/features/build_location_features.py"),
                "--interim-dir", str(interim), "--output-dir", str(root / "out"),
                "--report-dir", str(root / "reports"), "--mrt-mode", "historical",
                "--mrt-opening-dates", str(dates),
            ], capture_output=True, text=True, encoding="utf-8")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("80.00%", result.stdout)
            output = pd.read_csv(root / "out/train_location_features.csv")
            self.assertEqual(output.loc[0, "OPEN_MRT_COUNT_WITHIN_500M"], 1)
            self.assertEqual(output.loc[1, "OPEN_MRT_COUNT_WITHIN_500M"], 2)

    def test_missing_data_does_not_create_fake_report(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaisesRegex(FileNotFoundError, "clean_data.py"):
                run_pipeline(root / "missing", root / "out", root / "reports")
            self.assertFalse((root / "out").exists())
            self.assertFalse((root / "reports").exists())

    def test_rejects_output_inside_input_and_invalid_radii(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_fixture(root / "interim")
            with self.assertRaisesRegex(ValueError, "outside"):
                run_pipeline(root / "interim", root / "interim/out", root / "reports")
            with self.assertRaisesRegex(ValueError, "distinct positive"):
                run_pipeline(root / "interim", root / "out", root / "reports", (500, 500))


if __name__ == "__main__":
    unittest.main()
