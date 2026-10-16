"""Temporal selection and training isolation for bounded tree tuning."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.run_stage5 import feature_schema
from src.models.run_tree_tuning import (
    BASE_IDS, EARLIER, blend_menu, choose_locked, combine, load_reference,
    nested_one, search_grid, shortlist, split_inner, train_tree,
)


def fixture():
    dates = pd.date_range("2023-01-01", "2025-03-01", freq="MS").repeat(4)
    frame = pd.DataFrame({"RENT_APPROVAL_DATE": dates})
    cats, nums = feature_schema("housing")
    for column in cats:
        frame[column] = ["a", "b"] * (len(frame) // 2)
    for column in nums:
        frame[column] = (np.arange(len(frame)) % 11).astype(float)
    frame["MONTHLY_RENT"] = 2000 + np.arange(len(frame)) * 10
    return frame


class TreeTuningTests(unittest.TestCase):
    def test_grid_is_bounded_and_old_models_unchanged(self):
        grid = search_grid()
        self.assertEqual(len(grid), 8)
        self.assertEqual(grid[BASE_IDS["catboost"]]["cap"], 400)
        self.assertEqual(grid[BASE_IDS["lightgbm"]]["params"]["num_leaves"], 31)
        self.assertEqual(grid[BASE_IDS["catboost"]]["patience"], 40)

    def test_inner_is_strictly_earlier_and_last_three_months(self):
        fit, inner = split_inner(fixture())
        self.assertLess(fit.RENT_APPROVAL_DATE.max(), inner.RENT_APPROVAL_DATE.min())
        self.assertEqual(inner.RENT_APPROVAL_DATE.dt.to_period("M").nunique(), 3)
        with self.assertRaises(ValueError):
            split_inner(inner)

    def test_shortlist_uses_only_new_screen_candidates_per_family(self):
        grid = search_grid()
        rows = [dict(candidate=name, family=spec["family"], rmse=1 if name in BASE_IDS.values() else 10 + i)
            for i, (name, spec) in enumerate(grid.items())]
        selected = shortlist(pd.DataFrame(rows), grid)
        self.assertEqual(selected["catboost"], "catboost_d6_long")
        self.assertEqual(selected["lightgbm"], "lightgbm_31_slow")

    def test_blend_menu_keeps_original_and_valid_weights(self):
        menu = blend_menu(dict(catboost="new_cat", lightgbm="new_lgb"))
        self.assertEqual(len(menu), 16)
        self.assertIn({"catboost_original": 0.5, "lightgbm_original": 0.5}, menu.values())
        for weights in menu.values():
            self.assertAlmostEqual(sum(weights.values()), 1)
        np.testing.assert_allclose(combine(dict(a=np.array([1., 3.]), b=np.array([3., 5.])), dict(a=.25, b=.75)), [2.5, 4.5])
        with self.assertRaises(ValueError):
            combine({}, {"a": -.1, "b": 1.1})

    def test_selection_excludes_primary_and_uses_pooled_not_mean_rmse(self):
        rows = []
        for split, n, scores in [(EARLIER[0], 100, [1, 3]), (EARLIER[1], 1, [10, 3]), ("temporal", 100000, [1000, 0])]:
            for name, score in zip(["a", "b"], scores):
                rows.append(dict(split=split, candidate=name, validation_rows=n, rmse=score))
        selected, table = choose_locked(pd.DataFrame(rows))
        self.assertEqual(selected, "a")
        self.assertAlmostEqual(table.iloc[0].pooled_rmse, np.sqrt(200 / 101))

    def test_selection_rejects_incomplete_or_duplicate_folds(self):
        one = dict(split=EARLIER[0], candidate="a", validation_rows=10, rmse=5)
        for rows in [[one], [one, one]]:
            with self.assertRaises(ValueError):
                choose_locked(pd.DataFrame(rows))

    def test_unknown_categories_and_native_model_smoke(self):
        frame = fixture()
        query = frame.iloc[-4:].copy()
        query["TOWN"] = "unseen"
        for family in BASE_IDS:
            spec = search_grid()[BASE_IDS[family]]
            bundle = train_tree(spec, frame.iloc[:-4], 1, iterations=3)
            self.assertTrue(np.isfinite(bundle.predict(query)).all())
            self.assertNotIn("unseen", bundle.levels["TOWN"])

    def test_outer_labels_never_affect_rounds_or_predictions(self):
        frame = fixture()
        fit, query = frame.iloc[:-12], frame.iloc[-12:]
        for family in BASE_IDS:
            spec = {**search_grid()[BASE_IDS[family]], "cap": 4, "patience": 2}
            with tempfile.TemporaryDirectory() as left, tempfile.TemporaryDirectory() as right:
                pred, details = nested_one("tiny", spec, "test", fit, query, Path(left), 1)
                other, metadata = nested_one("tiny", spec, "test", fit, query.assign(MONTHLY_RENT=1e9), Path(right), 1)
                np.testing.assert_allclose(pred, other)
                self.assertEqual(details["iterations"], metadata["iterations"])
                self.assertLess(details["inner_fit_end"], details["inner_validation_start"])

    def test_reference_rejects_misaligned_rows(self):
        query = fixture().iloc[-4:]
        frame = pd.DataFrame(dict(source_row=query.index, actual=query.MONTHLY_RENT,
            catboost_housing=[1.] * 4, lightgbm_housing=[2.] * 4))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions_temporal.csv"
            frame.to_csv(path, index=False)
            self.assertEqual(set(load_reference(Path(tmp), "temporal", query)), set(BASE_IDS.values()))
            frame.iloc[::-1].to_csv(path, index=False)
            with self.assertRaises(ValueError):
                load_reference(Path(tmp), "temporal", query)


if __name__ == "__main__":
    unittest.main()
