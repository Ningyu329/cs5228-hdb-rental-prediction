"""Fit-only neural preprocessing, temporal selection, model serialization."""
import importlib.util
import tempfile
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from src.models.keras_rent import fit_schema, transform, recipe_schema, train, load_bundle, release, TIME_NUMS
from src.models.run_stage5 import feature_schema
from src.models.run_keras_experiments import ranking, specs


def houses():
    dates = pd.date_range("2021-01-01", periods=9, freq="MS").repeat(8)
    cats, nums = feature_schema("housing")
    data = pd.DataFrame({c: ["a", "b"] * (len(dates) // 2) for c in cats})
    for col in nums:
        data[col] = 1.
    data["RENT_APPROVAL_DATE"] = dates
    data["RENT_MONTH"] = dates.month
    data["RENT_MONTH_INDEX"] = dates.month
    data["FLOOR_AREA_SQM"] = 60 + np.arange(len(dates)) % 8 * 5
    data["MONTHLY_RENT"] = (1200 + data.FLOOR_AREA_SQM * 10 + dates.month * 20).astype(int)
    return data


class KerasTests(unittest.TestCase):
    def test_vocabulary_is_fit_only_and_unknowns_use_zero(self):
        data = houses()
        data.loc[0, "TOWN"] = "singleton"
        schema = fit_schema(data, "direct")
        query = data.iloc[:2].assign(TOWN="unknown")
        encoded = transform(schema, query)
        self.assertNotIn("unknown", schema["levels"]["TOWN"])
        self.assertNotIn("singleton", schema["levels"]["TOWN"])
        i = schema["cats"].index("TOWN")
        np.testing.assert_array_equal(encoded[f"category_{i}"], 0)

    def test_missing_constant_and_nonfinite_numeric_inputs(self):
        data = houses()
        data["FLOOR_AREA_SQM"] = np.nan
        schema = fit_schema(data, "direct")
        query = data.iloc[:2].copy()
        query["FLOOR_AREA_SQM"] = [np.inf, np.nan]
        x = transform(schema, query)["numeric"]
        self.assertTrue(np.isfinite(x).all())
        i = schema["nums"].index("FLOOR_AREA_SQM")
        np.testing.assert_array_equal(x[:, i + len(schema["nums"])], 1)
        self.assertTrue(all(s > 0 for s in schema["scales"]))

    def test_transform_ignores_labels_and_ids_without_mutation(self):
        data = houses()
        before = data.copy(deep=True)
        schema = fit_schema(data, "direct")
        original = transform(schema, data)
        altered = transform(schema, data.assign(MONTHLY_RENT=-999999., Id=12345, source_row=-1))
        for name in original:
            np.testing.assert_array_equal(original[name], altered[name])
        pd.testing.assert_frame_equal(data, before)

    def test_residual_schema_has_no_calendar_or_deterministic_age(self):
        cats, nums = recipe_schema("trend")
        self.assertFalse(TIME_NUMS.intersection(nums))
        self.assertNotIn("FLAT_AGE_BAND", cats)
        self.assertFalse({"MONTHLY_RENT", "Id", "source_row"}.intersection(cats + nums))
        with self.assertRaises(ValueError):
            fit_schema(houses(), "unknown")

    def test_early_stopping_rejects_nonfuture_inner_window(self):
        data = houses()
        with self.assertRaises(ValueError):
            train(data, "direct", 42, specs()["direct"], valid=data)

    def test_selection_uses_pooled_earlier_only_and_promotion_gate(self):
        rows = [dict(split=s, candidate=c, rows=n, rmse=r) for s, n in [("q3_2024", 10), ("q4_2024", 30)]
                for c, r in [("incumbent", 450.), ("direct_blend100", 449.5), ("trend_blend100", 452.)]]
        scores, pure, new, selected = ranking(rows + [dict(split="primary_2025", candidate="trend_blend100", rows=999, rmse=0.)])
        self.assertEqual(pure, "direct_blend100")
        self.assertEqual(selected, "incumbent")
        self.assertEqual(new, "direct_blend100")
        self.assertEqual(scores.iloc[0].pooled_rmse, 449.5)
        with self.assertRaises(ValueError):
            ranking(rows[1:])
        with self.assertRaises(ValueError):
            ranking(rows + rows[:1])

    @unittest.skipUnless(importlib.util.find_spec("keras") and importlib.util.find_spec("jax"), "Keras/JAX optional dependencies")
    def test_both_recipes_integer_labels_roundtrip_and_query_invariance(self):
        data = houses()
        fit, query = data.iloc[:48].copy(), data.iloc[48:].copy()
        for kind in ["direct", "trend"]:
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as d:
                spec = dict(specs()[kind], widths=[8], max_epochs=2, batch_size=32)
                bundle, detail = train(fit, kind, 42, spec, epochs=2, label="fixture")
                expected = bundle.predict(query)
                self.assertTrue(np.isfinite(expected).all())
                self.assertEqual(detail["best_epoch"], 2)
                bundle.save(Path(d))
                loaded = load_bundle(Path(d))
                np.testing.assert_allclose(loaded.predict(query), expected, atol=1e-5, rtol=0)
                np.testing.assert_allclose(loaded.predict(query.assign(MONTHLY_RENT=-999999., Id=99)), expected, atol=1e-5, rtol=0)
                self.assertTrue(np.isfinite(loaded.predict(query.assign(TOWN="unseen"))).all())
                self.assertEqual(bundle.response(fit).dtype, np.dtype("float32"))
                release(bundle)


if __name__ == "__main__":
    unittest.main()
