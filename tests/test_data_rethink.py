"""Raw alignment, public history cutoffs and structural inference contracts."""
import unittest
import tempfile
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from src.data.clean_data import clean_main_rows
from src.features.build_raw_views import verify_alignment, attach_raw_view, RAW_CATS, RAW_NUMS
from src.features.build_resale_history import snapshot, causal_resale, FEATURES, canonical_street, canonical_town
from src.models.hierarchical_rent import HierarchicalRent
from src.models.run_data_rethink import choose, Experiment


def houses():
    return pd.DataFrame(dict(RENT_APPROVAL_DATE=pd.to_datetime(["2023-01-01", "2023-02-01", "2023-03-01", "2023-04-01"]),
        TOWN="town", BLOCK=["1", "1", "2", "3"], STREET="street", FLAT_TYPE="3-room", FLAT_MODEL="model",
        FLOOR_AREA_SQM=[70., 70., 80., 75.], LEASE_COMMENCE_DATE=1990, LATITUDE=1.3, LONGITUDE=103.8,
        MONTHLY_RENT=[2000., 2100., 2500., 2300.]))


def history():
    data = houses().rename(columns={"FLOOR_AREA_SQM": "AREA"})
    data["MONTH"] = data.RENT_APPROVAL_DATE.dt.to_period("M").astype("int64")
    data["PSM"] = [3000., 4000., 5000., 6000.]
    data["HIGH_FLOOR"] = [0., 1., 0., 1.]
    return data


class RethinkTests(unittest.TestCase):
    def test_official_address_abbreviations_do_not_lose_matches(self):
        self.assertEqual(canonical_street("Bedok South Avenue 3"), canonical_street("BEDOK STH AVE 3"))
        self.assertEqual(canonical_street("Punggol Road"), canonical_street("PUNGGOL RD"))
        self.assertEqual(canonical_town("CENTRAL AREA"), "central")
        self.assertNotEqual(canonical_street("Road 1"), canonical_street("Road 11"))

    def test_raw_alignment_and_target_unchanged(self):
        raw = pd.DataFrame([dict(RENT_APPROVAL_DATE="2023-01", TOWN="town", BLOCK="1", STREET="STREET",
            FLAT_TYPE="3 room", FLAT_MODEL="model", FLOOR_AREA_SQM="70.0", FURNISHED="yes",
            LEASE_COMMENCE_DATE="1990", FEE="0.0", MONTHLY_RENT="2000")])
        clean, _ = clean_main_rows(raw.to_dict("records"), has_target=True)
        clean = pd.DataFrame(clean)
        self.assertTrue(all(r["aligned"] for r in verify_alignment(raw, clean)))
        joined = attach_raw_view(clean, raw)
        self.assertEqual(joined.RAW_STREET.iloc[0], "STREET")
        self.assertEqual(joined.RAW_FLAT_FORMAT_CHANGED.iloc[0], 1)
        pd.testing.assert_series_equal(joined.MONTHLY_RENT, clean.MONTHLY_RENT)
        bad = clean.copy()
        bad.loc[0, "BLOCK"] = "999"
        with self.assertRaises(ValueError):
            attach_raw_view(bad, raw)
        bad = clean.copy()
        bad.loc[0, "MONTHLY_RENT"] = "999"
        with self.assertRaises(ValueError):
            verify_alignment(raw, bad)

    def test_resale_lag_frozen_and_query_labels_irrelevant(self):
        data = houses()
        fit, query = data.iloc[:3], data.iloc[3:]
        a, b = causal_resale(fit, query, history(), lag=2)
        self.assertEqual(a.loc[2, "RESALE_TOWN_MEDIAN_PSM"], 3000.)
        self.assertEqual(b.loc[3, "RESALE_TOWN_MEDIAN_PSM"], 3000.)
        self.assertEqual(b.loc[3, "RESALE_STALENESS_MONTHS"], 3)
        changed = history()
        changed.loc[1:, "PSM"] = 999999.
        _, same = causal_resale(fit, query.assign(MONTHLY_RENT=-999), changed, lag=2)
        pd.testing.assert_frame_equal(b, same)
        self.assertNotIn("MONTHLY_RENT", b)
        self.assertTrue(a.loc[:1, "RESALE_TOWN_MEDIAN_PSM"].isna().all())

    def test_history_window_unknown_keys_and_boundaries(self):
        h = history()
        query = houses().iloc[-1:].assign(BLOCK="unseen")
        cutoff = int(h.MONTH.max())
        a = snapshot(h, query, cutoff, window=1)
        self.assertEqual(a.RESALE_TOWN_COUNT.iloc[0], 1)
        self.assertEqual(a.RESALE_BLOCK_COUNT.iloc[0], 0)
        self.assertTrue(np.isnan(a.RESALE_BLOCK_MEDIAN_PSM.iloc[0]))
        with self.assertRaises(ValueError):
            causal_resale(houses(), houses().iloc[-1:], h)

    def test_structural_model_ignores_query_label_and_unknown_categories(self):
        data = houses()
        model = HierarchicalRent().fit(data.iloc[:3])
        query = data.iloc[3:].copy()
        expected = model.predict(query)
        np.testing.assert_allclose(model.predict(query.assign(MONTHLY_RENT=-999999., Id=900)), expected)
        self.assertTrue(np.isfinite(model.predict(query.assign(TOWN="unseen", FLAT_TYPE="other"))).all())
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "model.joblib"
            joblib.dump(model, path)
            np.testing.assert_allclose(joblib.load(path).predict(query), expected)

    def test_structural_damping_is_continuous_and_bounded(self):
        data = houses()
        model = HierarchicalRent().fit(data)
        query = pd.concat([data.iloc[-1:]] * 3, ignore_index=True)
        query["RENT_APPROVAL_DATE"] = pd.to_datetime(["2023-04-01", "2023-05-01", "2099-01-01"])
        self.assertTrue(np.isfinite(model.matrix(query).data).all())
        self.assertLess(np.max(np.abs(model.matrix(query).toarray())), 10.)

    def test_selection_never_uses_primary_and_respects_gain_gate(self):
        rows = [dict(split=s, candidate=c, rows=10, rmse=r) for s in ["q3_2024", "q4_2024"]
                for c, r in [("incumbent", 450.), ("challenger", 449.)]]
        _, selected = choose(rows + [dict(split="primary_2025", candidate="challenger", rows=999, rmse=0.)])
        self.assertEqual(selected, "incumbent")
        rows = [dict(r, rmse=440.) if r["candidate"] == "challenger" else r for r in rows]
        _, selected = choose(rows)
        self.assertEqual(selected, "challenger")

    def test_integer_rent_labels_complete_residual_fit_and_cache_reload(self):
        from tests.test_rmse_improvement import fixture
        data = fixture()
        data["MONTHLY_RENT"] = data.MONTHLY_RENT.astype("int64")
        fit, query = data.iloc[:-12].copy(), data.iloc[-12:].copy()
        with tempfile.TemporaryDirectory() as directory:
            exp = Experiment.__new__(Experiment)
            exp.cache, exp.threads, exp.details = Path(directory), 1, []
            exp.frames = lambda split: (fit.copy(), query.copy())
            first = exp.component("fixture", "hierarchy_residual_lgb")
            np.testing.assert_allclose(first, exp.component("fixture", "hierarchy_residual_lgb"))
            self.assertTrue(np.isfinite(first).all())
            self.assertEqual(len(first), len(query))

    def test_long_horizon_selection_ignores_latest_origin(self):
        from src.models.select_long_horizon import EARLY, CHECK, CANDIDATES, select
        rows = [dict(split=s, candidate=c, rows=10, rmse=450. if c == "housing_lgb" else 440.)
                for s in EARLY for c in CANDIDATES]
        ranking, selected = select(rows)
        changed = rows + [dict(split=CHECK, candidate=selected, rows=999999, rmse=999999.)]
        self.assertEqual(select(changed), (ranking, selected))


if __name__ == "__main__":
    unittest.main()
