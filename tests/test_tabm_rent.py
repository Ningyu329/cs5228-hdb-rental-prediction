"""TabM branch loss, input cutoffs and local model restoration contracts."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.models.tabm_rent import independent_mse, fit_tabm_schema, tensors, make_network, train, load_bundle
from src.models.run_tabm_experiments import specs
from src.models.run_stage5 import feature_schema


def houses():
    cats, nums = feature_schema("housing")
    dates = pd.date_range("2021-01-01", periods=9, freq="MS").repeat(8)
    data = pd.DataFrame({c: ["a", "b"] * (len(dates) // 2) for c in cats})
    for col in nums:
        data[col] = 1.
    data["FLOOR_AREA_SQM"] = 60 + np.arange(len(dates)) % 8 * 5
    data["RENT_APPROVAL_DATE"] = dates
    data["RENT_MONTH"] = dates.month
    data["MONTHLY_RENT"] = (1200 + data.FLOOR_AREA_SQM * 10 + dates.month * 20).astype(int)
    return data


@unittest.skipUnless(importlib.util.find_spec("tabm") and importlib.util.find_spec("torch"), "TabM dependencies optional")
class TabMTests(unittest.TestCase):
    def test_branch_loss_is_not_loss_of_mean_prediction(self):
        import torch
        prediction = torch.tensor([[-1., 1.]], requires_grad=True)
        loss = independent_mse(prediction, torch.tensor([0.]))
        self.assertEqual(loss.item(), 1.)
        loss.backward()
        self.assertTrue(torch.all(prediction.grad != 0))
        with self.assertRaises(ValueError):
            independent_mse(prediction.ravel(), torch.tensor([0.]))

    def test_bins_use_only_training_predictors_and_constants_are_not_binned(self):
        data = houses()
        schema = fit_tabm_schema(data, "direct", 4)
        self.assertNotIn(schema["nums"].index("LEASE_COMMENCE_DATE"), schema["ple_indices"])
        self.assertEqual(schema, fit_tabm_schema(data.assign(MONTHLY_RENT=-999999., Id=123), "direct", 4))
        self.assertTrue(all(len(edges) >= 2 and np.all(np.diff(edges) > 0) for edges in schema["bins"]))
        self.assertEqual(len(schema["raw_indices"]) + len(schema["ple_indices"]), 2 * len(schema["nums"]))

    def test_unknown_categories_and_missing_values_remain_finite(self):
        data = houses()
        schema = fit_tabm_schema(data, "direct", 4)
        numeric, categorical = tensors(schema, data.iloc[:2].assign(TOWN="unseen", FLOOR_AREA_SQM=np.nan), "cpu")
        self.assertTrue(np.isfinite(numeric.numpy()).all())
        self.assertEqual(int(categorical[:, schema["cats"].index("TOWN")].sum()), 0)

    def test_official_backbone_keeps_separate_branch_outputs(self):
        data = houses()
        schema = fit_tabm_schema(data, "direct", 4)
        spec = dict(specs()["direct"], k=4, n_blocks=1, d_block=8, d_embedding=4)
        net = make_network(schema, spec)
        result = net(*tensors(schema, data.iloc[:3], "cpu"))
        self.assertEqual(tuple(result.shape), (3, 4))
        self.assertEqual(net.backbone.k, 4)

    def test_inner_window_cannot_overlap_fit_or_have_fixed_epochs(self):
        data = houses()
        with self.assertRaises(ValueError):
            train(data, "direct", 42, specs()["direct"], valid=data)
        with self.assertRaises(ValueError):
            train(data.iloc[:48], "direct", 42, specs()["direct"], valid=data.iloc[48:], epochs=2)

    def test_integer_labels_both_targets_save_load_and_query_invariance(self):
        data = houses()
        fit, query = data.iloc[:48].copy(), data.iloc[48:].copy()
        for kind in ["direct", "trend"]:
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as d:
                spec = dict(specs()[kind], k=4, n_blocks=1, d_block=8, d_embedding=4,
                            n_bins=4, batch_size=32, max_epochs=2, patience=1)
                bundle, detail = train(fit, kind, 42, spec, epochs=2, device="cpu", label="fixture")
                prediction = bundle.predict(query)
                bundle.save(Path(d))
                restored = load_bundle(Path(d))
                np.testing.assert_allclose(restored.predict(query), prediction, atol=1e-5, rtol=0)
                np.testing.assert_allclose(restored.predict(query.assign(MONTHLY_RENT=-999999., Id=12345)), prediction, atol=1e-5, rtol=0)
                self.assertTrue(np.isfinite(prediction).all())
                self.assertEqual(bundle.response(fit).dtype, np.dtype("float32"))
                self.assertEqual(detail["best_epoch"], 2)


if __name__ == "__main__":
    unittest.main()
