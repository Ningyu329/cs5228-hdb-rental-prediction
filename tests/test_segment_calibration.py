"""Contracts for past-only group corrections and SSE accounting."""
import unittest

import numpy as np
import pandas as pd

from src.features.segment_calibration import fit_correction, correction_values, segments, available_macro


def fixture():
    frame = pd.DataFrame({"RENT_APPROVAL_DATE": pd.date_range("2023-01-01", periods=6, freq="MS").repeat(20)})
    frame["TOWN"] = ["a", "b"] * 60
    frame["FLAT_TYPE"] = ["3-room", "4-room"] * 60
    frame["REGION"] = "region"
    frame["FLOOR_AREA_SQM"] = 80.
    frame["FLAT_AGE"] = 20.
    frame["BLOCK_ADDRESS"] = frame.TOWN + "|1|road"
    frame["MONTHLY_RENT"] = 2500.
    frame["prediction"] = 2400.
    return frame


class SegmentTests(unittest.TestCase):
    def test_constant_correction_is_shrunk_and_unseen_group_falls_back(self):
        frame = fixture()
        model = fit_correction(frame, ["TOWN"], shrink=60)
        query = frame.iloc[:2].copy()
        query["RENT_APPROVAL_DATE"] = pd.Timestamp("2023-07-01")
        np.testing.assert_allclose(correction_values(model, query), 50)
        query["TOWN"] = "unknown"
        np.testing.assert_allclose(correction_values(model, query), 0)

    def test_query_labels_are_irrelevant_and_same_month_is_rejected(self):
        frame = fixture()
        model = fit_correction(frame, ["TOWN"])
        query = frame.iloc[:2].copy()
        with self.assertRaises(ValueError):
            correction_values(model, query)
        query["RENT_APPROVAL_DATE"] = pd.Timestamp("2024-01-01")
        first = correction_values(model, query)
        query["MONTHLY_RENT"] = -100000.
        np.testing.assert_array_equal(first, correction_values(model, query))

    def test_tiny_segments_get_zero_not_noisy_corrections(self):
        frame = fixture().iloc[:5]
        model = fit_correction(frame, ["TOWN"], min_rows=30)
        query = frame.copy()
        query["RENT_APPROVAL_DATE"] = pd.Timestamp("2024-01-01")
        np.testing.assert_allclose(correction_values(model, query), 0)

    def test_trend_is_damped_clipped_and_does_not_read_future_targets(self):
        frame = fixture()
        month = pd.to_datetime(frame.RENT_APPROVAL_DATE).dt.month
        frame["MONTHLY_RENT"] = 2400 + month * 10
        model = fit_correction(frame, [], shrink=0, trend=True)
        query = frame.iloc[:1].copy()
        query["RENT_APPROVAL_DATE"] = pd.Timestamp("2023-07-01")
        self.assertAlmostEqual(correction_values(model, query)[0], 60 + 60 * (1 - np.exp(-1/6)))
        query["RENT_APPROVAL_DATE"] = pd.Timestamp("2100-01-01")
        self.assertAlmostEqual(correction_values(model, query)[0], 120)
        frame["MONTHLY_RENT"] = 100000.
        clipped = fit_correction(frame, [], shrink=0)
        self.assertEqual(correction_values(clipped, query)[0], 200)

    def test_sse_contributions_sum_once_within_every_partition(self):
        frame = fixture()
        table = segments(frame, frame.prediction.to_numpy(), frame.iloc[:30])
        for _, group in table.groupby("partition"):
            self.assertEqual(group.rows.sum(), len(frame))
            self.assertAlmostEqual(group.sse_share.sum(), 1)
            self.assertAlmostEqual(group.sse.sum(), 10000 * len(frame))
            self.assertTrue((group.bias_sse_share <= group.sse_share + 1e-12).all())

    def test_available_macro_joins_same_month_without_forward_filling(self):
        frame = fixture().iloc[[0, -1]].copy()
        monthly = pd.DataFrame({"COE_A_PREMIUM": [10., 20.]}, index=pd.PeriodIndex(["2023-01", "2023-06"], freq="M"))
        actual = available_macro(frame, monthly)
        np.testing.assert_array_equal(actual.COE_A_PREMIUM, [10., 20.])
        frame["MONTHLY_RENT"] = 100000.
        np.testing.assert_array_equal(available_macro(frame, monthly).COE_A_PREMIUM, [10., 20.])
        frame["RENT_APPROVAL_DATE"] = pd.Timestamp("2024-01-01")
        self.assertTrue(available_macro(frame, monthly).COE_A_PREMIUM.isna().all())


if __name__ == "__main__":
    unittest.main()
