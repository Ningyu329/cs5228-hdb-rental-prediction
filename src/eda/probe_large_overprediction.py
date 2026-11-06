"""Fixed diagnostic classifier: can observable inputs flag >1000 overpricing?

This is not a rental-price model or a submitted routing rule. All training
labels come from past expanding-origin OOF errors, not in-sample residuals.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eda.analyze_error_mechanisms import attach_history, WINDOWS
from src.models.run_stage5 import load_training, feature_schema, FittedModel

PAST = {"q3_2024":["cal_q1_2024","cal_q2_2024"],"q4_2024":["cal_q2_2024","q3_2024"],"primary_2025":["q3_2024","q4_2024"]}
ALL_WINDOWS = {**WINDOWS,"cal_q1_2024":("2024-01","2024-03"),"cal_q2_2024":("2024-04","2024-06")}
EXTRA = ["CONTROL_PREDICTION","HISTORY_N","ALL_HISTORY_N","EXACT_HOUSE_N","HISTORY_MEAN","HISTORY_MEDIAN","HISTORY_STD","HISTORY_ADJUSTED_STD","HISTORY_Q10","HISTORY_Q90"]
PARAMS = dict(n_estimators=300,learning_rate=.04,num_leaves=15,min_child_samples=100,
    reg_lambda=10.,colsample_bytree=.9,random_state=42,n_jobs=3,verbosity=-1,
    deterministic=True,force_col_wise=True)


def feature_matrix(fit: pd.DataFrame, query: pd.DataFrame, numeric: list[str], cats: list[str]):
    levels = {c:fit[c].fillna("missing").astype(str).unique().tolist() for c in cats}
    bundle=FittedModel("lightgbm",None,cats,numeric,levels)
    if set(["MONTHLY_RENT","ERROR","SQUARED_ERROR","TAIL5","source_row"]) & set(bundle.features):
        raise ValueError("Target-derived or identity columns cannot be inputs")
    return bundle.matrix(fit),bundle.matrix(query)


def top_alert_metrics(actual: np.ndarray, probability: np.ndarray) -> dict:
    n=max(1,int(np.ceil(len(actual)*.05)))
    order=np.argsort(-probability,kind="stable")[:n]
    positives=int(np.asarray(actual)[order].sum())
    return dict(rows=len(actual),positive_rows=int(actual.sum()),prevalence=float(actual.mean()),
        roc_auc=float(roc_auc_score(actual,probability)),average_precision=float(average_precision_score(actual,probability)),
        alert_rows=n,alert_precision=positives/n,alert_recall=positives/int(actual.sum()),
        alert_lift=(positives/n)/float(actual.mean()))


def run(output: Path):
    output=output.resolve()
    if output.exists():
        raise FileExistsError("Use a fresh probe output directory")
    output.mkdir(parents=True)
    source=ROOT/"data/processed/train_location_features.csv"
    train=load_training(source)
    reference=ROOT/"reports/experiments/rmse_improvement_v1"
    original=json.loads((reference/"experiment_manifest.json").read_text())
    lock=json.loads((reference/"selection_locked.json").read_text())
    cache=ROOT/"models/rmse_improvement_cache"/original["run_id"]
    prepared={}
    oof_provenance=[]
    for split,(start,end) in ALL_WINDOWS.items():
        months=train.RENT_APPROVAL_DATE.dt.to_period("M")
        fit=train.loc[months.lt(pd.Period(start))]
        query=train.loc[months.between(pd.Period(start),pd.Period(end))]
        prediction=np.zeros(len(query))
        for name,weight in lock["weights"].items():
            with np.load(cache/f"pred_{split}_{name}.npz") as saved:
                np.testing.assert_array_equal(saved["source_row"],query.index)
                prediction+=weight*saved["prediction"]
        features=attach_history(fit,query)
        features["CONTROL_PREDICTION"]=prediction
        features["_OVER1000"]=(prediction-query.MONTHLY_RENT.to_numpy()>1000).astype(int)
        prepared[split]=features
        oof_provenance.append(dict(split=split,fit_end=f"{fit.RENT_APPROVAL_DATE.max():%Y-%m}",
            query_start=f"{query.RENT_APPROVAL_DATE.min():%Y-%m}",query_end=f"{query.RENT_APPROVAL_DATE.max():%Y-%m}",rows=len(query)))
    cats,numeric=feature_schema("housing")
    rows=[]
    for split in WINDOWS:
        fit=pd.concat([prepared[n] for n in PAST[split]])
        query=prepared[split]
        assert fit.RENT_APPROVAL_DATE.max()<query.RENT_APPROVAL_DATE.min()
        assert not fit.index.isin(query.index).any()
        for variant,extra in [("housing",[]),("housing_history_control",EXTRA)]:
            a,b=feature_matrix(fit,query,numeric+extra,cats)
            # Fixed 300 rounds. Outer labels never select parameters or rounds.
            model=LGBMClassifier(**PARAMS)
            model.fit(a,fit._OVER1000,categorical_feature=cats)
            probability=model.predict_proba(b)[:,1]
            rows.append(dict(split=split,variant=variant,training_rows=len(fit),**top_alert_metrics(query._OVER1000.to_numpy(),probability)))
    result=dict(params=PARAMS,target="OOF prediction - actual rent > 1000",
        features=dict(categorical=cats,numeric=numeric,extra=EXTRA),past_windows=PAST,
        no_model_selection=True,no_rent_model_changed=True,no_submission=True,no_test_data_read=True,
        provenance=oof_provenance,results=rows,
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [source,Path(__file__),ROOT/"src/eda/analyze_error_mechanisms.py",reference/"experiment_manifest.json",reference/"selection_locked.json"]},
        limits=["fixed simple classifier failure does not establish absence of predictive information",
            "all diagnostic windows have previously been inspected", "top-5% alert quota is fixed without true query labels"])
    (output/"probe.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    pd.DataFrame(rows).to_csv(output/"probe_results.csv",index=False)
    lines=["# Identifiability of Large Overpredictions","",
        "A fixed LightGBM binary diagnostic predicts which rows the original model will overpredict by more than 1000; rent predictions are not modified. Training targets come from strictly earlier OOF errors.", "",
        "| Window | Inputs | Share overpredicted >1000 | ROC-AUC | AP | Precision in top-5% risk | Recall of positives | Lift over random |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        lines.append(f"| {row['split']} | {row['variant']} | {row['prevalence']:.2%} | {row['roc_auc']:.3f} | {row['average_precision']:.3f} | {row['alert_precision']:.2%} | {row['alert_recall']:.2%} | {row['alert_lift']:.2f}x |")
    lines += ["", "AP is average precision. The expected AP of a random ranking is close to the positive share; the top-5% risk set is a fixed alert budget, so no true validation label is used to pick a threshold.", "",
        "History-extended inputs include the original prediction, the mean/median/std/quantiles of same-block same-flat-type rents over the past twelve months, and history coverage. Each historical OOF quarter is built only from earlier training data.", "",
        "This experiment only tests this fixed diagnostic; it does not prove that no feature or model can identify cheap rentals, and the probabilities are not used to correct rent predictions. True rent bins, errors and row ids are never inputs.", ""]
    (output/"probe_report.md").write_text("\n".join(lines),encoding="utf-8")
    print(pd.DataFrame(rows).round(4).to_string(index=False),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=ROOT/"reports/experiments/overprediction_probe_v1")
    run(parser.parse_args().output_dir)
