"""Independently reconcile diagnosis aggregates and a fresh probe replication."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
INPUTS=["RENT_APPROVAL_DATE","TOWN","BLOCK","STREET","FLAT_TYPE","FLAT_MODEL","FLOOR_AREA_SQM","FURNISHED","LEASE_COMMENCE_DATE","FEE"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(output: Path, probe: Path, replicate: Path):
    diagnosis=json.loads((output/"diagnosis.json").read_text())
    for file,expected in diagnosis["source_sha256"].items():
        assert sha(ROOT/file)==expected,file
    train=pd.read_csv(ROOT/"data/processed/train_location_features.csv")
    raw=pd.read_csv(ROOT/"data/raw/train.csv")
    checked_rows,partitions=0,0
    for split,expected in diagnosis["summary"].items():
        saved=pd.read_csv(ROOT/f"reports/experiments/segment_v1/predictions_{split}.csv")
        query=train.iloc[saved.source_row].copy()
        y=query.MONTHLY_RENT.to_numpy()
        value=saved.locked_448_control.to_numpy()
        np.testing.assert_array_equal(y,saved.actual)
        error=value-y
        sse=np.dot(error,error)
        np.testing.assert_allclose(expected["rmse"],np.sqrt(sse/len(y)),atol=1e-9)
        np.testing.assert_allclose(expected["bias"],error.mean(),atol=1e-9)
        assert expected["error_over_1000"]==int((error>1000).sum())
        assert expected["error_under_minus1000"]==int((error < -1000).sum())
        np.testing.assert_allclose(expected["actual_below_2000_sse_share"],(error[y<=2000]**2).sum()/sse)
        for key,source in [("normalized_identical_inputs",query),("raw_identical_inputs",raw.iloc[saved.source_row])]:
            identifiers=source.groupby(INPUTS,dropna=False).ngroup().to_numpy(dtype=int)
            counts=np.bincount(identifiers)
            sums=np.bincount(identifiers,weights=y)
            mean=sums[identifiers]/counts[identifiers]
            within=np.dot(y-mean,y-mean)
            np.testing.assert_allclose(expected[key]["within_sse"],within,rtol=1e-12)
            assert expected[key]["repeated_rows"]==int(counts[counts>1].sum())
            np.testing.assert_allclose(expected[key]["retrospective_constant_rmse"],np.sqrt(within/len(y)),atol=1e-9)
        table=pd.read_csv(output/f"error_mechanisms_{split}.csv")
        for _,group in table.groupby("partition"):
            assert group.rows.sum()==len(query)
            np.testing.assert_allclose(group.sse.sum(),sse,rtol=1e-12)
            np.testing.assert_allclose(group.sse_share.sum(),1.,atol=1e-12)
            partitions+=1
        descending=np.sort(error**2)[::-1]
        for fraction,data in expected["tails"].items():
            n=int(np.ceil(float(fraction)*len(query)))
            remaining=sse-descending[:n].sum()
            np.testing.assert_allclose(data["hypothetical_rmse_if_tail_errors_zero"],np.sqrt(remaining/len(query)),atol=1e-9)
        checked_rows+=len(query)
    first=json.loads((probe/"probe.json").read_text())
    second=json.loads((replicate/"probe.json").read_text())
    for file,expected in first["source_sha256"].items():
        assert sha(ROOT/file)==expected,file
    assert first["params"]==second["params"]
    assert first["features"]==second["features"]
    assert first["results"]==second["results"]
    assert first["no_model_selection"] and first["no_rent_model_changed"]
    for row in first["results"]:
        true_positive=row["alert_precision"]*row["alert_rows"]
        np.testing.assert_allclose(true_positive,round(true_positive),atol=1e-10)
        np.testing.assert_allclose(row["alert_recall"],true_positive/row["positive_rows"],atol=1e-12)
        np.testing.assert_allclose(row["prevalence"],row["positive_rows"]/row["rows"],atol=1e-12)
        assert row["alert_rows"]==int(np.ceil(row["rows"]*.05))
    for row in first["provenance"]:
        assert row["fit_end"]<row["query_start"]<=row["query_end"]
    submitted=ROOT/"submissions/submission_rmse448_trend25_20261019.csv"
    receipt=json.loads((ROOT/"reports/submissions/rmse448_20261019/submission_manifest.json").read_text())
    assert sha(submitted)==receipt["output_sha256"]
    result=dict(passed=True,validation_rows_checked=checked_rows,partitions_checked=partitions,
        probe_scores_reproduced=len(first["results"]),probe_replication_directory=str(replicate.relative_to(ROOT)),
        submitted_prediction_unchanged=True,rent_models_changed=False,
        inspector_sha256=sha(Path(__file__)),cleaner_sha256=sha(ROOT/"src/data/clean_data.py"))
    (output/"output_validation.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2))
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=ROOT/"reports/experiments/error_diagnosis_v1")
    parser.add_argument("--probe-dir",type=Path,default=ROOT/"reports/experiments/overprediction_probe_v1")
    parser.add_argument("--replicate-dir",type=Path,default=ROOT/"tmp/overprediction_probe_reproduction_20261019")
    args=parser.parse_args()
    audit(args.output_dir,args.probe_dir,args.replicate_dir)
