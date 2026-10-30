import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "models"))
import run_te_experiments as te  # noqa: E402


def frame(n=60, seed=0):
    rng = np.random.default_rng(seed)
    blocks = [f"b{i}" for i in range(6)]
    df = pd.DataFrame({
        "ADDRESS_KEY": rng.choice(blocks, n), "FLAT_TYPE": rng.choice(["3-room", "4-room"], n),
        "FLOOR_AREA_SQM": rng.choice([70.0, 90.0], n), "FLAT_MODEL": "improved",
        "TOWN": rng.choice(["t1", "t2"], n), "STREET": rng.choice(["s1", "s2"], n),
        "SUBZONE": "z", "MONTH_ID": rng.integers(600, 603, n),
        "MONTHLY_RENT": rng.integers(2000, 4000, n).astype(float),
    })
    coords = {b: (1.30 + 0.002 * i, 103.80 + 0.002 * i) for i, b in enumerate(blocks)}
    df["LATITUDE"] = df["ADDRESS_KEY"].map(lambda b: coords[b][0])
    df["LONGITUDE"] = df["ADDRESS_KEY"].map(lambda b: coords[b][1])
    return df


class TargetEncodingTests(unittest.TestCase):
    def setUp(self):
        te.OPTIONS.update(stats=True, knn=True, recent_months=0)

    def tearDown(self):
        te.OPTIONS.update(stats=False, knn=False)

    def test_query_labels_never_used(self):
        fit, query = frame(seed=1), frame(20, seed=2)
        target = fit["MONTHLY_RENT"].to_numpy() / 3000
        a = te.encode_groups(fit, target, query, 1.0)
        b = te.encode_groups(fit, target, query.assign(MONTHLY_RENT=-1.0), 1.0)
        pd.testing.assert_frame_equal(a, b)
        self.assertEqual(len(a), len(query))

    def test_knn_excludes_own_block(self):
        fit = frame(seed=3)
        target = np.where(fit["ADDRESS_KEY"] == "b0", 100.0, 1.0)
        query = fit[fit["ADDRESS_KEY"] == "b0"].head(3)
        feats = te.knn_features(fit, target, query)
        self.assertTrue(np.allclose(feats["knn_te"].dropna(), 1.0))

    def test_volume_counts_same_month_rows(self):
        df = frame(seed=4)
        vol = te.volume_features(df)
        expected = df.groupby(["ADDRESS_KEY", "MONTH_ID"])["MONTH_ID"].transform("size")
        self.assertTrue((vol["vol_addr"] == expected).all())
        self.assertTrue(np.allclose(vol.groupby(df["MONTH_ID"])["vol_town"].apply(
            lambda s: s.groupby(df.loc[s.index, "TOWN"]).first().sum()), 1.0))


if __name__ == "__main__":
    unittest.main()
