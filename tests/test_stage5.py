"""Causal macro timing, tree category handling and time extrapolation tests."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.features.build_stage5_features import COE_FEATURES, STOCK_FEATURES, add_housing_features, aggregate_macro, attach_macro
from src.models.run_stage5 import FittedModel, cache_identity, feature_schema, fit_candidate, fit_tree, group_permutation, select_candidate, spline_pipeline


def macro_fixtures():
    coe = pd.DataFrame([
        {"YEAR_MONTH": month, "CATEGORY": category, "QUOTA_PREMIUM": premium, "QUOTA": 10, "BIDS_RECEIVED": 20}
        for month in ["2024-10", "2024-11", "2024-12", "2025-01", "2025-02"]
        for category in ["a", "b", "c", "e"] for premium in [100, 200]
    ])
    stock = pd.DataFrame([
        {"DATE": date, "SYMBOL": symbol, "CLOSE": close}
        for symbol in ["A", "B"]
        for date, close in [("2024-10-01", 10), ("2024-10-31", 10), ("2024-11-01", 11), ("2024-11-29", 12), ("2024-12-02", 13), ("2024-12-31", 14), ("2025-01-02", 15), ("2025-01-31", 16)]
    ])
    return coe, stock


class Stage5Tests(unittest.TestCase):
    def test_cache_identity_covers_implementation_but_not_progress(self):
        keys = ["environment", "source_sha256", "tree_iterations_cap", "threads", "random_seed", "catboost", "lightgbm", "early_stopping", "macro", "feature_sets", "constant_fee_values", "constant_furnished_values", "selection_rule", "implementation_sha256"]
        manifest = {key: "fixed" for key in keys}
        original = cache_identity(manifest)
        with_progress = {**manifest, "selected_candidate": "catboost_main", "workers": 8}
        self.assertEqual(original, cache_identity(with_progress))
        self.assertNotEqual(original, cache_identity({**manifest, "implementation_sha256": "changed"}))

    def test_coe_round_aggregation_and_stock_monthly_returns(self):
        monthly, audit = aggregate_macro(*macro_fixtures())
        self.assertEqual(monthly.loc[pd.Period("2024-11"), "COE_A_PREMIUM"], 150)
        self.assertEqual(monthly.loc[pd.Period("2024-11"), "COE_A_BID_QUOTA_RATIO"], 2)
        self.assertAlmostEqual(monthly.loc[pd.Period("2024-11"), "STOCK_EQ_WEIGHT_RETURN"], 0.2)
        self.assertEqual(monthly.loc[pd.Period("2024-11"), "STOCK_SYMBOL_COUNT"], 2)
        self.assertEqual(audit["stock_missing_close_rows"], 0)
        self.assertEqual(len(monthly.columns), 12)

    def test_macro_lag_and_frozen_forecast_cutoff(self):
        monthly, _ = aggregate_macro(*macro_fixtures())
        frame = pd.DataFrame({"RENT_APPROVAL_DATE": ["2025-01", "2025-02", "2025-03"]})
        rolling = attach_macro(frame, monthly)
        frozen = attach_macro(frame, monthly, pd.Timestamp("2024-12-01"))
        self.assertEqual(rolling.MACRO_SOURCE_MONTH.tolist(), ["2024-11", "2024-12", "2025-01"])
        self.assertEqual(frozen.MACRO_SOURCE_MONTH.tolist(), ["2024-11"] * 3)

    def test_future_market_values_cannot_change_frozen_validation_features(self):
        coe, stock = macro_fixtures()
        monthly, _ = aggregate_macro(coe, stock)
        mutated_coe = coe.copy()
        mutated_coe.loc[mutated_coe.YEAR_MONTH.ge("2024-12"), "QUOTA_PREMIUM"] = 999999
        mutated_stock = stock.copy()
        mutated_stock.loc[mutated_stock.DATE.ge("2024-12-01"), "CLOSE"] = 999999
        mutated, _ = aggregate_macro(mutated_coe, mutated_stock)
        frame = pd.DataFrame({"RENT_APPROVAL_DATE": ["2025-01", "2025-02", "2025-03"]})
        original = attach_macro(frame, monthly, pd.Timestamp("2024-12-01"))
        changed = attach_macro(frame, mutated, pd.Timestamp("2024-12-01"))
        pd.testing.assert_frame_equal(original, changed)

    def test_missing_prices_do_not_use_backfill_and_duplicate_dates_fail(self):
        coe, stock = macro_fixtures()
        stock.loc[0, "CLOSE"] = np.nan
        _, audit = aggregate_macro(coe, stock)
        self.assertEqual(audit["stock_missing_close_rows"], 1)
        with self.assertRaisesRegex(ValueError, "unique"):
            aggregate_macro(coe, pd.concat([stock, stock.iloc[[1]]]))

    def test_housing_transforms_use_only_observed_attributes(self):
        frame = pd.DataFrame({"RENT_APPROVAL_DATE": ["2025-01", "2025-02"], "FLOOR_AREA_SQM": [90, 150], "FLAT_TYPE": ["3-room", "executive"], "FLAT_AGE": [25, 45], "TOWN": ["Town A", "Town B"], "BLOCK": ["01", "2"], "STREET": ["Road A", "Road B"]})
        out = add_housing_features(frame)
        self.assertEqual(out.loc[0, "AREA_PER_ROOM"], 30)
        self.assertTrue(pd.isna(out.loc[1, "AREA_PER_ROOM"]))
        self.assertEqual(out.loc[0, "BLOCK_ADDRESS"], "town a|01|road a")
        self.assertEqual(frame.columns.size + 7, out.columns.size)

    def test_selection_never_uses_primary_or_snapshot_scores(self):
        rows = []
        for split in ["temporal_backtest_1", "temporal_backtest_2", "temporal"]:
            for candidate, scheme, score in [("catboost_housing", "housing", 500), ("lightgbm_housing", "housing", 510), ("catboost_mrt_snapshot", "mrt_snapshot", 1)]:
                if split == "temporal" and candidate == "lightgbm_housing":
                    score = 1
                rows.append({"split": split, "candidate": candidate, "scheme": scheme, "family": candidate.split("_")[0], "rmse": score, "validation_rows": 10})
        selected, table = select_candidate(pd.DataFrame(rows))
        self.assertEqual(selected, "catboost_housing")
        self.assertTrue(table.windows.eq(2).all())

    def test_constant_spline_extrapolation(self):
        model, cats, nums = spline_pipeline()
        frame = pd.DataFrame({column: ["x"] * 20 for column in cats})
        for column in nums:
            frame[column] = np.arange(20, dtype=float) if column == "RENT_MONTH_INDEX" else 1.0
        model.fit(frame, np.arange(20) * 100 + 2000)
        query = pd.concat([frame.iloc[[-1]], frame.iloc[[-1]]], ignore_index=True)
        query.loc[1, "RENT_MONTH_INDEX"] = 1000
        self.assertAlmostEqual(*model.predict(query), places=8)

    def test_tree_smoke_tests_with_unseen_categories(self):
        fit = pd.DataFrame({"TOWN": ["a", "b"] * 30, "AREA": np.arange(60), "MONTHLY_RENT": 2000 + np.arange(60) * 20})
        query = pd.DataFrame({"TOWN": ["unseen", "a"], "AREA": [15, 30]})
        for family in ["catboost", "lightgbm"]:
            bundle = fit_tree(family, fit, query, ["TOWN"], ["AREA"], 5, 1)
            self.assertTrue(np.isfinite(bundle.predict(query)).all())
            if family == "lightgbm":
                self.assertNotIn("unseen", bundle.levels["TOWN"])

    def test_mixed_group_permutation_preserves_numeric_dtypes(self):
        fit = pd.DataFrame({"SUBZONE": ["a", "b"] * 30, "LATITUDE": np.arange(60, dtype=float), "MONTHLY_RENT": 2000 + np.arange(60) * 20})
        bundle = fit_tree("lightgbm", fit, fit, ["SUBZONE"], ["LATITUDE"], 5, 1)
        importance = group_permutation([bundle], fit, 40)
        self.assertEqual(importance.iloc[0]["group"], "hdb")
        self.assertTrue(np.isfinite(importance.rmse_increase_mean).all())

    def test_macro_features_only_enter_the_intended_schemes(self):
        self.assertTrue(set(COE_FEATURES).isdisjoint(feature_schema("amenities")[1]))
        self.assertTrue(set(COE_FEATURES).issubset(feature_schema("coe")[1]))
        self.assertTrue(set(STOCK_FEATURES).isdisjoint(feature_schema("coe")[1]))
        self.assertTrue(set(STOCK_FEATURES).issubset(feature_schema("market")[1]))

    def test_outer_query_labels_cannot_influence_tree_rounds_or_predictions(self):
        dates = pd.date_range("2024-01-01", "2025-03-01", freq="MS").repeat(4)
        cats, nums = feature_schema("main")
        frame = pd.DataFrame({"RENT_APPROVAL_DATE": dates})
        for column in cats:
            frame[column] = ["a", "b"] * (len(frame) // 2)
        for column in nums:
            frame[column] = 1.0 + np.arange(len(frame)) % 7
        frame["MONTHLY_RENT"] = 2000 + np.arange(len(frame)) * 10
        fit = frame.iloc[:-12]
        query = frame.iloc[-12:]
        changed = query.assign(MONTHLY_RENT=999999.0)
        monthly, _ = aggregate_macro(*macro_fixtures())
        for family in ["catboost", "lightgbm"]:
            _, prediction, metadata, _ = fit_candidate(family, "main", fit, query, monthly, 5, 1)
            _, alternate, other, _ = fit_candidate(family, "main", fit, changed, monthly, 5, 1)
            self.assertEqual(metadata["iterations"], other["iterations"])
            np.testing.assert_allclose(prediction, alternate)
            self.assertEqual(metadata["inner_validation_end"], "2024-12")
            self.assertLess(metadata["inner_fit_end"], metadata["inner_validation_start"])


if __name__ == "__main__":
    unittest.main()
