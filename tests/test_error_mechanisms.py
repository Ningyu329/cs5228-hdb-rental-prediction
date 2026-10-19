"""Diagnostic contracts; no source data or rental models are changed."""
import unittest

import numpy as np
import pandas as pd

from src.eda.analyze_error_mechanisms import collisions, attach_history, make_view, summarize_partition
from src.eda.probe_large_overprediction import feature_matrix, top_alert_metrics


def history_fixture():
    fit=pd.DataFrame({"RENT_APPROVAL_DATE":pd.date_range("2023-01-01",periods=6,freq="MS").repeat(2)})
    for c in ["TOWN","BLOCK","STREET","FLAT_TYPE","FLAT_MODEL"]:
        fit[c]="a"
    fit["FLOOR_AREA_SQM"]=80.
    fit["LEASE_COMMENCE_DATE"]=1980
    fit["FLAT_AGE"]=43
    fit["MONTHLY_RENT"]=[2000.,3000.]*6
    query=fit.iloc[:2].copy()
    query["RENT_APPROVAL_DATE"]=pd.Timestamp("2023-07-01")
    return fit,query


class DiagnosisTests(unittest.TestCase):
    def test_empirical_collision_floor_is_not_generalization_error(self):
        frame=pd.DataFrame({"x":["a","a","b"],"MONTHLY_RENT":[100.,300.,900.]})
        result=collisions(frame,["x"])
        self.assertEqual(result["repeated_rows"],2)
        self.assertEqual(result["conflicting_groups"],1)
        self.assertAlmostEqual(result["retrospective_constant_rmse"],np.sqrt(20000/3))
        frame["arbitrary_id"]=[1,2,3]
        # Adding an uninformative identifier trivially lowers the retrospective
        # floor. Thus raw formatting splits alone do not prove useful signal.
        self.assertEqual(collisions(frame,["x","arbitrary_id"])["retrospective_constant_rmse"],0)

    def test_historical_covariates_ignore_query_labels(self):
        fit,query=history_fixture()
        first=attach_history(fit,query)
        query["MONTHLY_RENT"]=-100000
        second=attach_history(fit,query)
        names=[c for c in first if c.startswith("HISTORY_") or c.endswith("_N")]
        pd.testing.assert_frame_equal(first[names],second[names])
        self.assertTrue(first.HISTORY_N.eq(12).all())
        self.assertTrue(first.EXACT_HOUSE_N.eq(12).all())
        with self.assertRaises(ValueError):
            attach_history(fit,fit.iloc[:1])

    def test_unknown_house_uses_zero_coverage_and_missing_spread(self):
        fit,query=history_fixture()
        query["BLOCK"]="new"
        out=attach_history(fit,query)
        self.assertTrue(out.HISTORY_N.eq(0).all())
        self.assertTrue(out.EXACT_HOUSE_N.eq(0).all())
        self.assertTrue(out.HISTORY_STD.isna().all())

    def test_partition_preserves_all_rows_and_squared_error(self):
        fit,query=history_fixture()
        view=make_view(fit,query,np.array([2200.,3100.]))
        expected=200**2+100**2
        for field in ["ACTUAL_RENT_BIN","EXACT_HOUSE_BIN","HISTORICAL_SPREAD_BIN"]:
            table=summarize_partition(view,field)
            self.assertEqual(table.rows.sum(),2)
            self.assertAlmostEqual(table.sse.sum(),expected)
            self.assertAlmostEqual(table.sse_share.sum(),1)

    def test_probe_features_exclude_label_and_error_columns(self):
        fit,query=history_fixture()
        a,b=feature_matrix(fit,query,["FLOOR_AREA_SQM"],["TOWN"])
        query["MONTHLY_RENT"]+=100000
        _,changed=feature_matrix(fit,query,["FLOOR_AREA_SQM"],["TOWN"])
        pd.testing.assert_frame_equal(b,changed)
        self.assertNotIn("MONTHLY_RENT",a)
        with self.assertRaises(ValueError):
            feature_matrix(fit,query,["FLOOR_AREA_SQM","MONTHLY_RENT"],["TOWN"])

    def test_fixed_alert_quota_does_not_depend_on_true_labels(self):
        labels=np.zeros(20,dtype=int)
        labels[:2]=1
        probability=np.arange(20,0,-1)/20
        result=top_alert_metrics(labels,probability)
        self.assertEqual(result["alert_rows"],1)
        self.assertEqual(result["alert_precision"],1)
        self.assertEqual(result["alert_recall"],.5)
        self.assertEqual(result["average_precision"],1)
        self.assertEqual(result["alert_lift"],10)


if __name__=="__main__":
    unittest.main()
