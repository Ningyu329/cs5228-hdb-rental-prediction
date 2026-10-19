"""Causality, unit-loss identity and earlier-only selection contracts."""
import tempfile
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from src.features.build_neighborhood_features import (
    ALL_FEATURES, area_values, causal_features, snapshot,
)
from src.models.run_neighborhood_experiments import response, choose


def fixture():
    rows = pd.DataFrame(dict(RENT_APPROVAL_DATE=pd.to_datetime([
        "2023-01-01", "2023-01-01", "2023-02-01", "2023-03-01"]),
        TOWN="town", BLOCK=["1", "1", "2", "1"], STREET="street", FLAT_TYPE="3-room",
        FLAT_MODEL="model", FLOOR_AREA_SQM=100., LEASE_COMMENCE_DATE=1990,
        LATITUDE=[1., 1., 1.01, 1.], LONGITUDE=103., MONTHLY_RENT=[1000., 3000., 5000., 9000.]))
    return rows


class NeighborhoodTests(unittest.TestCase):
    def test_entire_current_month_is_excluded_and_future_labels_irrelevant(self):
        data = fixture()
        first, query = causal_features(data.iloc[:3], data.iloc[3:])
        self.assertTrue(first.loc[:1, "PAST_BLOCK_MEAN_RATE"].isna().all())
        self.assertEqual(first.loc[2, "NEAR_16_MEAN_RATE"], 20.)
        self.assertEqual(query.loc[3, "PAST_EXACT_MEAN_RATE"], 20.)
        changed = data.copy()
        changed.loc[2:, "MONTHLY_RENT"] = -100000.
        other, changed_query = causal_features(changed.iloc[:3], changed.iloc[3:])
        pd.testing.assert_frame_equal(first[ALL_FEATURES], other[ALL_FEATURES])
        # Modifying past fit labels legitimately changes future summaries.
        self.assertNotEqual(query.loc[3, "NEAR_16_MEAN_RATE"], changed_query.loc[3, "NEAR_16_MEAN_RATE"])
        changed = data.iloc[3:].copy()
        changed["MONTHLY_RENT"] = -100000.
        _, same = causal_features(data.iloc[:3], changed)
        pd.testing.assert_frame_equal(query, same)
        self.assertNotIn("MONTHLY_RENT", query)

    def test_nearest_reference_is_blocks_not_repeated_transactions(self):
        data = fixture()
        result = snapshot(data.iloc[:3], data.iloc[3:])
        self.assertEqual(result.loc[3, "NEAR_16_BLOCKS"], 2)
        self.assertEqual(result.loc[3, "NEAR_16_MEAN_RATE"], 35.)
        self.assertEqual(result.loc[3, "NEAR_16_STD_RATE"], 15.)
        self.assertAlmostEqual(result.loc[3, "NEAR_16_FARTHEST_KM"], 1.1119508023, places=7)
        self.assertEqual(result.loc[3, "PAST_BLOCK_COUNT"], 2)
        self.assertEqual(result.loc[3, "PAST_BLOCK_Q10_RATE"], 12)

    def test_unseen_flat_type_or_bad_coordinates_do_not_borrow_wrong_type(self):
        data = fixture()
        query = data.iloc[3:].copy()
        query["FLAT_TYPE"] = "new"
        self.assertEqual(snapshot(data.iloc[:3], query).loc[3, "NEAR_16_BLOCKS"], 0)
        query = data.iloc[3:].copy()
        query["LATITUDE"] = np.nan
        self.assertTrue(np.isnan(snapshot(data.iloc[:3], query).loc[3, "NEAR_16_MEAN_RATE"]))

    def test_frozen_forecast_does_not_update_from_query_months(self):
        data = fixture()
        query = pd.concat([data.iloc[3:], data.iloc[3:].assign(RENT_APPROVAL_DATE=pd.Timestamp("2024-01-01"))], ignore_index=True)
        _, out = causal_features(data.iloc[:3], query)
        np.testing.assert_allclose(out[ALL_FEATURES].iloc[0], out[ALL_FEATURES].iloc[1], equal_nan=True)
        with self.assertRaises(ValueError):
            causal_features(data, data.iloc[:1])

    def test_cache_ignores_query_labels_but_invalidates_changed_history(self):
        data = fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            _, first = causal_features(data.iloc[:3], data.iloc[3:], path)
            count = len(list(path.glob("*.npz")))
            _, same = causal_features(data.iloc[:3], data.iloc[3:].assign(MONTHLY_RENT=-1), path)
            self.assertEqual(count, len(list(path.glob("*.npz"))))
            pd.testing.assert_frame_equal(first, same)
            _, other = causal_features(data.iloc[:3].assign(MONTHLY_RENT=4000), data.iloc[3:], path)
            self.assertGreater(len(list(path.glob("*.npz"))), count)
            self.assertNotEqual(first.loc[3, "PAST_BLOCK_MEAN_RATE"], other.loc[3, "PAST_BLOCK_MEAN_RATE"])

    def test_area_squared_loss_is_dollar_squared_loss_up_to_constant(self):
        frame = fixture()
        frame["FLOOR_AREA_SQM"] = [50., 70., 100., 120.]
        y, weights = response(frame, True)
        rate_prediction = np.array([18., 20., 35., 40.])
        areas = area_values(frame)
        self.assertAlmostEqual(np.sum(weights * (y-rate_prediction)**2) * np.mean(areas**2),
                               np.sum((frame.MONTHLY_RENT-areas*rate_prediction)**2))
        for invalid in [0., -1., np.nan, np.inf]:
            with self.assertRaises(ValueError):
                area_values(frame.assign(FLOOR_AREA_SQM=invalid))

    def test_selection_requires_material_gain_and_quarter_stability(self):
        rows = [dict(split=s, candidate=n, rows=100, rmse=r) for s in ["q3_2024", "q4_2024"]
                for n, r in [("incumbent", 450.), ("tiny", 449.9), ("good", 448.)]]
        rows += [dict(split="primary_2025", candidate="tiny", rows=100, rmse=100.)]
        _, selected, _ = choose(rows)
        self.assertEqual(selected, "good")
        rows = [r for r in rows if r["candidate"] != "good"]
        self.assertEqual(choose(rows)[1], "incumbent")
        rows += [dict(split="q3_2024", candidate="unstable", rows=100, rmse=440.),
                 dict(split="q4_2024", candidate="unstable", rows=100, rmse=452.)]
        self.assertEqual(choose(rows)[1], "incumbent")


if __name__ == "__main__":
    unittest.main()
