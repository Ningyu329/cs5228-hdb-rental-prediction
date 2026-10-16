"""Check fair validation splits, training-only preprocessing and report outputs."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.run_location_ablation import (
    LABELS, VARIANTS, add_comparisons, feature_schema, run_experiments, validation_splits,
)
from src.models.train_baseline import CATEGORICAL_FEATURES, NUMERIC_FEATURES, build_ridge_pipeline


def toy_training() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", "2025-03-01", freq="MS").repeat(4)
    n = len(dates)
    df = pd.DataFrame({"RENT_APPROVAL_DATE": dates})
    all_cats, all_nums = set(), set()
    for variant in VARIANTS:
        cats, nums = feature_schema(variant)
        all_cats.update(cats)
        all_nums.update(nums)
    for column in all_cats:
        df[column] = ["a" if i % 2 else "b" for i in range(n)]
    for column in all_nums:
        df[column] = 1 + np.arange(n) % 5
    df["RENT_YEAR"] = dates.year
    df["RENT_MONTH"] = dates.month
    df["RENT_MONTH_INDEX"] = (dates.year - 2021) * 12 + dates.month
    df["MONTHLY_RENT"] = 2000 + np.arange(n) * 5
    return df


class LocationAblationTests(unittest.TestCase):
    def test_baseline_schema_and_parameters_remain_unchanged(self):
        model = build_ridge_pipeline()
        self.assertEqual(model.named_steps["preprocess"].transformers[0][2], CATEGORICAL_FEATURES)
        self.assertEqual(model.named_steps["preprocess"].transformers[1][2], NUMERIC_FEATURES)
        self.assertEqual(feature_schema("main_only"), (CATEGORICAL_FEATURES, NUMERIC_FEATURES))
        ridge = model.named_steps["model"]
        self.assertEqual(ridge.alpha, 10.0)
        self.assertEqual(ridge.tol, 1e-4)
        self.assertEqual(ridge.solver, "lsqr")
        self.assertFalse(ridge.fit_intercept)

    def test_time_splits_do_not_train_on_later_rows(self):
        df = toy_training()
        splits = validation_splits(df, 3, 2, True)
        self.assertEqual([name for name, _, _ in splits], ["temporal", "temporal_backtest_1", "temporal_backtest_2", "random"])
        for _, fit, valid in splits[:3]:
            self.assertLess(fit.RENT_APPROVAL_DATE.max(), valid.RENT_APPROVAL_DATE.min())
            self.assertFalse(set(fit.index) & set(valid.index))
            self.assertEqual(valid.RENT_APPROVAL_DATE.dt.to_period("M").nunique(), 3)
        self.assertEqual(splits[0][2].RENT_APPROVAL_DATE.min(), pd.Timestamp("2025-01-01"))
        self.assertEqual(splits[1][2].RENT_APPROVAL_DATE.min(), pd.Timestamp("2024-10-01"))
        self.assertEqual(splits[2][2].RENT_APPROVAL_DATE.min(), pd.Timestamp("2024-07-01"))
        self.assertEqual(len(splits[-1][2]), 12)
        self.assertEqual(len(set(splits[-1][1].index) | set(splits[-1][2].index)), len(df))

    def test_preprocessing_fits_only_on_training_values(self):
        fit = pd.DataFrame({"TOWN": ["a", "a"], "AREA": [10.0, 20.0]})
        valid = pd.DataFrame({"TOWN": ["unseen", "unseen"], "AREA": [np.nan, 99999.0]})
        model = build_ridge_pipeline(["TOWN"], ["AREA"])
        model.fit(fit, [1000.0, 2000.0])
        prediction = model.predict(valid)
        numeric = model.named_steps["preprocess"].named_transformers_["numeric"]
        self.assertEqual(numeric.named_steps["impute"].statistics_[0], 15.0)
        self.assertEqual(numeric.named_steps["scale"].mean_[0], 15.0)
        self.assertTrue(np.isfinite(prediction).all())

    def test_improvements_compare_to_the_same_split(self):
        frame = pd.DataFrame([
            {"split": "temporal", "variant": "main_only", "rmse": 600.0},
            {"split": "temporal", "variant": "hdb", "rmse": 570.0},
            {"split": "random", "variant": "main_only", "rmse": 500.0},
            {"split": "random", "variant": "hdb", "rmse": 480.0},
        ])
        out = add_comparisons(frame)
        self.assertEqual(out.rmse_improvement.tolist(), [0, 30, 0, 20])
        self.assertEqual(out.improvement_pct.tolist(), [0, 5, 0, 4])

    def test_unknown_or_insufficient_splits_fail_clearly(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            validation_splits(toy_training(), 0, 0, False)
        with self.assertRaisesRegex(ValueError, "Not enough"):
            validation_splits(toy_training(), 3, 6, False)

    def test_complete_small_run_writes_metrics_features_and_private_predictions(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            train = root / "train.csv"
            toy_training().to_csv(train, index=False)
            provenance = root / "location_manifest.json"
            provenance.write_text(json.dumps({"mrt_mode": "snapshot"}), encoding="utf-8")
            output = root / "reports"
            results, manifest = run_experiments(train, output, provenance, ["main_only", "hdb", "hdb_mrt"], 3, 1, True, 562.59)
            self.assertEqual(len(results), 9)
            self.assertTrue(np.isfinite(results.rmse).all())
            self.assertTrue(manifest["best_primary_variant_without_mrt"] in {"main_only", "hdb"})
            self.assertFalse(manifest["feature_sets"]["hdb"]["uses_mrt"])
            self.assertTrue(manifest["feature_sets"]["hdb_mrt"]["uses_mrt"])
            predictions = pd.read_csv(output / "primary_validation_predictions.csv")
            self.assertEqual(len(predictions), 12)
            original = toy_training()
            expected = original.loc[predictions.source_row, "MONTHLY_RENT"].to_numpy()
            np.testing.assert_array_equal(predictions.actual, expected)
            groups = pd.read_csv(output / "group_errors.csv")
            self.assertEqual(set(groups.group_field), {"TOWN", "FLAT_TYPE", "RENT_APPROVAL_DATE", "REGION"})
            report = (output / "location_ablation_report.md").read_text(encoding="utf-8")
            self.assertIn("snapshot", report)
            self.assertIn("562.5900", report)
            self.assertEqual(set(LABELS), set(VARIANTS))


if __name__ == "__main__":
    unittest.main()
