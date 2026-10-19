"""XGBoost encodings, inverse targets, serialization and temporal selection."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from tests.test_rmse_improvement import fixture

AVAILABLE = importlib.util.find_spec("xgboost") is not None
if AVAILABLE:
    from src.models.run_xgboost_experiments import grid, train, load_bundle, ranking, prepare


@unittest.skipUnless(AVAILABLE, "Install requirements-xgboost.txt to test XGBoost")
class XGBoostTests(unittest.TestCase):
    def small(self, name):
        spec = dict(grid()[name], cap=12, child=1., l2=1.)
        frame = fixture()
        return spec, frame.iloc[:-12].copy(), frame.iloc[-12:].copy()

    def test_native_categories_use_fit_levels_and_unseen_values_are_missing(self):
        spec, fit, query = self.small("native_d6")
        bundle = train(spec, fit, threads=1, rounds=4)
        query["TOWN"] = "unknown"
        matrix = bundle.matrix(query)
        self.assertTrue(matrix.TOWN.isna().all())
        self.assertEqual(matrix.TOWN.cat.categories.tolist(), bundle.levels["TOWN"])
        self.assertNotIn("unknown", bundle.levels["TOWN"])
        self.assertTrue(np.isfinite(bundle.predict(query)).all())

    def test_onehot_keeps_real_numeric_zero_explicit(self):
        spec, fit, query = self.small("onehot_d6")
        fit["MAX_FLOOR"] = 0.
        bundle = train(spec, fit, threads=1, rounds=4)
        query["MAX_FLOOR"] = 0.
        matrix = bundle.matrix(query)
        offset = len(bundle.encoder.get_feature_names_out())
        column = offset + bundle.nums.index("MAX_FLOOR")
        self.assertEqual(matrix[:, column].nnz, len(query))
        np.testing.assert_array_equal(matrix[:, column].data, 0.)
        query["TOWN"] = "unknown"
        self.assertEqual(bundle.matrix(query).shape, matrix.shape)

    def test_query_targets_and_ids_never_enter_matrices_or_predictions(self):
        for name in ["native_d6", "onehot_d6", "native_unit_d6", "native_trend_d6"]:
            with self.subTest(name=name):
                spec, fit, query = self.small(name)
                bundle = train(spec, fit, threads=1, rounds=4)
                _, xquery, _, _, _ = prepare(fit, query, spec)
                first = bundle.predict(xquery)
                query["MONTHLY_RENT"], query["Id"] = -100000., 12345
                _, changed, _, _, _ = prepare(fit, query, spec)
                np.testing.assert_array_equal(first, bundle.predict(changed))
                self.assertNotIn("MONTHLY_RENT", bundle.cats+bundle.nums)
                self.assertNotIn("Id", bundle.cats+bundle.nums)

    def test_saved_native_and_onehot_models_restore_all_target_transforms(self):
        for name in ["native_d6", "onehot_d6", "native_unit_d6", "native_trend_d6"]:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                spec, fit, query = self.small(name)
                bundle = train(spec, fit, threads=1, rounds=4)
                bundle.save(Path(directory))
                restored = load_bundle(Path(directory))
                np.testing.assert_array_equal(bundle.predict(query), restored.predict(query))

    def test_unit_weights_are_used_for_early_stopping(self):
        spec, fit, query = self.small("native_unit_d6")
        bundle = train(spec, fit, threads=1, valid=query)
        y, weights = bundle.response(query)
        rate = bundle.model.predict(bundle.matrix(query))
        expected = np.sqrt(np.average((rate-y)**2, weights=weights))
        self.assertAlmostEqual(bundle.model.best_score, expected, places=4)

    def test_ranking_ignores_primary_and_requires_both_quarters(self):
        rows = [dict(split=s, candidate=n, rows=10, rmse=r)
                for s in ["q3_2024", "q4_2024"] for n, r in [
                    ("incumbent", 448.), ("a_blend100", 449.), ("a_blend25", 447.),
                    ("b_blend100", 440.), ("b_blend25", 441.)]]
        rows += [dict(split="primary_2025", candidate="a_blend100", rows=10000, rmse=0.)]
        table, pure, mixed = ranking(rows)
        self.assertEqual((pure, mixed), ("b_blend100", "b_blend100"))
        self.assertFalse(table.below_300.any())
        with self.assertRaises(ValueError):
            ranking(rows[1:])


if __name__ == "__main__":
    unittest.main()
