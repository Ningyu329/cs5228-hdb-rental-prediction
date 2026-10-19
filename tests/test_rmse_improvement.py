"""Temporal leakage, frozen forecast histories, trend and selection contracts."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.build_temporal_features import historical_features, history_columns, fit_trend, trend_values
from src.models.run_rmse_improvement import Spec, Experiment, prepare, sample_weights, score_choice, select_training
from src.models.run_stage5 import feature_schema


def fixture():
    dates = pd.date_range("2021-01-01", "2024-06-01", freq="MS").repeat(4)
    frame = pd.DataFrame({"RENT_APPROVAL_DATE": dates})
    cats, nums = feature_schema("housing")
    for column in cats:
        frame[column] = ["a", "b"] * (len(frame) // 2)
    for column in nums:
        frame[column] = np.arange(len(frame)) % 11 + 1.0
    frame["MONTHLY_RENT"] = 2000 + np.arange(len(frame)) * 5
    return frame


class ImprovementTests(unittest.TestCase):
    def test_history_excludes_own_month_and_all_later_labels(self):
        frame = fixture()
        fit, query = frame.iloc[:-12].copy(), frame.iloc[-12:].copy()
        left, right = historical_features(fit, query)
        cutoff = pd.Timestamp("2023-01-01")
        changed = fit.copy()
        changed.loc[changed.RENT_APPROVAL_DATE.ge(cutoff), "MONTHLY_RENT"] += 100000
        mutated, _ = historical_features(changed, query)
        pd.testing.assert_frame_equal(left.loc[left.RENT_APPROVAL_DATE.le(cutoff), history_columns(6)],
            mutated.loc[mutated.RENT_APPROVAL_DATE.le(cutoff), history_columns(6)])
        self.assertTrue(left.loc[left.RENT_APPROVAL_DATE.eq(fit.RENT_APPROVAL_DATE.min()), history_columns(6)[0]].isna().all())

    def test_forecast_snapshot_is_frozen_and_query_labels_are_unused(self):
        frame = fixture()
        fit, query = frame.iloc[:-12], frame.iloc[-12:].copy()
        _, first = historical_features(fit, query)
        changed = query.copy()
        changed["MONTHLY_RENT"] = -999999
        _, second = historical_features(fit, changed)
        pd.testing.assert_frame_equal(first[history_columns(6)], second[history_columns(6)])
        # Same attributes in each month retain one frozen set of statistics.
        np.testing.assert_allclose(first.iloc[0][history_columns(6)].astype(float), first.iloc[8][history_columns(6)].astype(float))
        with self.assertRaises(ValueError):
            historical_features(fit, fit.iloc[[-1]])

    def test_history_unknown_group_uses_past_prior_not_future_targets(self):
        frame = fixture()
        fit, query = frame.iloc[:-12], frame.iloc[-12:].copy()
        query["TOWN"] = "new"
        _, out = historical_features(fit, query)
        last_six = fit.loc[fit.RENT_APPROVAL_DATE.gt(fit.RENT_APPROVAL_DATE.max() - pd.DateOffset(months=6))]
        np.testing.assert_allclose(out.HIST_TOWN_FLAT_6M_MEAN, last_six.MONTHLY_RENT.mean())
        self.assertTrue(out.HIST_TOWN_FLAT_6M_COUNT.eq(0).all())

    def test_training_window_and_recency_weights(self):
        frame = fixture()
        recent = select_training(frame, Spec("lightgbm", window=24))
        self.assertEqual(recent.RENT_APPROVAL_DATE.dt.to_period("M").nunique(), 24)
        weights = sample_weights(frame, Spec("lightgbm", half_life=12))
        self.assertAlmostEqual(weights.mean(), 1)
        self.assertAlmostEqual(weights[-1] / weights[-49], 2)

    def test_trend_damps_future_growth_and_never_reads_query_target(self):
        frame = fixture()
        fit, query = frame.iloc[:-12], frame.iloc[-12:].copy()
        model = fit_trend(fit, "damped12")
        first = trend_values(model, query)
        query["MONTHLY_RENT"] = 1000000
        np.testing.assert_allclose(first, trend_values(model, query))
        far = query.iloc[[0]].copy()
        far["RENT_APPROVAL_DATE"] = pd.Timestamp("2050-01-01")
        self.assertLessEqual(trend_values(model, far)[0], model["level"] + model["slope"] * 6 + 1e-8)
        np.testing.assert_allclose(trend_values(fit_trend(fit, "flat3"), query), np.repeat(fit_trend(fit, "flat3")["level"], len(query)))

    def test_prepare_strips_query_target_and_time_fields_for_trend(self):
        frame = fixture()
        train, query, cats, nums, trend = prepare(frame.iloc[:-12], frame.iloc[-12:], Spec("lightgbm", trend="flat3"))
        self.assertNotIn("MONTHLY_RENT", query)
        for column in ["RENT_YEAR", "RENT_MONTH_INDEX", "BUILDING_AGE", "FLAT_AGE"]:
            self.assertNotIn(column, nums)
        self.assertNotIn("FLAT_AGE_BAND", cats)
        np.testing.assert_allclose(train.groupby("RENT_APPROVAL_DATE").MONTHLY_RENT.mean(), 0, atol=1e-8)

    def test_selection_is_pooled_and_ignores_primary(self):
        rows = []
        for split, n, scores in [("q3_2024", 100, [1, 3]), ("q4_2024", 1, [10, 3]), ("primary_2025", 10000, [999, 0])]:
            for name, score in zip(["a", "b"], scores):
                rows.append(dict(split=split, candidate=name, rmse=score, validation_rows=n))
        ranked = score_choice(rows, ["a", "b"])
        self.assertEqual(ranked.iloc[0].candidate, "a")
        with self.assertRaises(ValueError):
            score_choice(rows, ["a", "missing"])

    def test_native_residual_fit_reloads_and_outer_labels_are_unused(self):
        frame = fixture()
        fit, query = frame.iloc[:-12], frame.iloc[-12:]
        for recipe in [Spec("catboost", cap=5), Spec("lightgbm", history=6), Spec("lightgbm", trend="flat3")]:
            with tempfile.TemporaryDirectory() as directory:
                engine = Experiment.__new__(Experiment)
                engine.cache = Path(directory)
                engine.threads = 1
                first = engine._fit(fit, query, recipe, rounds=3)
                changed = query.copy()
                changed["MONTHLY_RENT"] += 1000000
                # Clear cached model so equality is not just cache reuse.
                for path in engine.cache.iterdir():
                    path.unlink()
                second = engine._fit(fit, changed, recipe, rounds=3)
                _, features, _, _, _ = prepare(fit, query, recipe)
                a = first["bundle"].model.predict(first["bundle"].matrix(features))
                b = second["bundle"].model.predict(second["bundle"].matrix(features))
                np.testing.assert_allclose(a, b)
                reloaded = engine._fit(fit, query, recipe, rounds=3)
                np.testing.assert_allclose(a, reloaded["bundle"].model.predict(reloaded["bundle"].matrix(features)))

    def test_nested_component_rounds_never_depend_on_outer_targets(self):
        frame = fixture()
        fit, query = frame.iloc[:-12], frame.iloc[-12:].copy()
        predictions, rounds = [], []
        with tempfile.TemporaryDirectory() as directory:
            for index, delta in enumerate([0, 1000000]):
                changed = query.copy()
                changed["MONTHLY_RENT"] += delta
                engine = Experiment.__new__(Experiment)
                engine.cache = Path(directory) / str(index)
                engine.cache.mkdir()
                engine.output = engine.cache
                engine.threads = 1
                engine.predictions, engine.specs = {}, {}
                engine.splits = {"outer": (fit, changed)}
                spec = Spec("catboost", cap=5)
                predictions.append(engine.component("outer", spec))
                import json
                rounds.append(json.loads((engine.output / f"fit_outer_{spec.name}.json").read_text())["iterations"])
        np.testing.assert_allclose(*predictions)
        self.assertEqual(*rounds)


if __name__ == "__main__":
    unittest.main()
