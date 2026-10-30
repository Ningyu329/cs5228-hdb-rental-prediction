"""Assemble the target-encoding submission from saved test predictions.

Steps (all on saved per-variant test predictions written by
``run_te_experiments.py --export``):
1. Average the four LightGBM feature variants.
2. Blend 75% with 25% of the incumbent trend-blend submission (weight chosen
   on the pooled 2024 Q3/Q4 backtests).
3. Apply the market-level adjustment ``level * (1 + trend * centred_month)``.
   The level and trend were estimated from public-leaderboard probes, not from
   local validation; see reports/submissions/te_trend_20261030/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
VARIANTS = ["te_v2_property", "te_v3_stats", "te_v3_knn", "te_v5_volume"]
INCUMBENT = ROOT / "submissions/submission_rmse448_trend25_20261019.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--te-weight", type=float, default=0.75)
    ap.add_argument("--level", type=float, default=1.017)
    ap.add_argument("--trend", type=float, default=0.002, help="relative change per month from the test mean month")
    ap.add_argument("--output", default=str(ROOT / "submissions/submission_te_avg4_blend75_trendprobe_20261029.csv"))
    args = ap.parse_args()

    incumbent = pd.read_csv(INCUMBENT)
    preds = []
    for v in VARIANTS:
        p = pd.read_csv(ROOT / f"reports/experiments/{v}/te_test_predictions.csv")
        if not (p["Id"].to_numpy() == incumbent["Id"].to_numpy()).all() or not np.isfinite(p["Predicted"]).all():
            raise ValueError(f"Bad predictions for {v}")
        preds.append(p["Predicted"].to_numpy())
    blend = args.te_weight * np.mean(preds, axis=0) + (1 - args.te_weight) * incumbent["Predicted"].to_numpy()

    test = pd.read_csv(ROOT / "data/raw/test.csv")
    month = pd.PeriodIndex(test["RENT_APPROVAL_DATE"], freq="M").asi8.astype(float)
    factor = args.level + args.trend * (month - month.mean())
    out = incumbent[["Id"]].assign(Predicted=blend * factor)
    out.to_csv(args.output, index=False)
    digest = hashlib.sha256(Path(args.output).read_bytes()).hexdigest()
    print(json.dumps({"output": args.output, "sha256": digest, "mean": float(out["Predicted"].mean()),
                      "factor_min": float(factor.min()), "factor_max": float(factor.max())}))


if __name__ == "__main__":
    main()
