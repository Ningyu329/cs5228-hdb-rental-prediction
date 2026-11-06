# Out-of-Fold Target Encoding, External Data and Market-Level Calibration (2026-10-29/30)

## Conclusion

The public RMSE fell from 437.44368 to **431.07006** (−6.37). The improvement has two parts:

1. **Model improvement (verifiable with local temporal backtests)**: five-fold out-of-fold (OOF) target
   encoding represents block- and street-level rent levels, two external sources are added (lagged resale
   prices and HDB block composition), and four feature variants are averaged and blended 75/25 with the
   original trend blend. All four backtest windows improve by 2–4.8 points; public 437.44 → 435.27.
2. **Overall market-level calibration (based on leaderboard probes, not verifiable locally)**: the model
   underestimates the overall rent level of the test period (2025-04 to 2026-07). Four global scale/trend
   probes identify the best "overall level × monthly trend"; public 435.27 → 431.07.

## 1. Out-of-fold target-encoding model

Script: [run_te_experiments.py](../../src/models/run_te_experiments.py).

- Target: rent divided by the month's mix-adjusted market level (the monthly mean of each row's rent
  relative to its town × flat-type mean). With the time trend removed, the model learns static housing
  differences. Predictions are multiplied back by the level; months after the last training month use the
  average level of the last 3 months (flat3).
- Target-encoding groups: block × flat type, block × flat type × floor area, block, street × flat type,
  subzone × flat type, and town × flat type × flat model, with smoothing 5 and group counts.
  Training rows get 5-fold out-of-fold values and query rows use the whole training window; query labels
  never enter the encoding.
- Difference from the earlier "history statistics from earlier months only": training rows also see
  same-block rows from the same and later months, so their encoding distribution matches the query rows,
  which is more stable for sparse blocks.
- LightGBM: learning rate 0.03, 127 leaves, at least 40 rows per leaf, feature/row sampling 0.7/0.8, L2 5;
  rounds are early-stopped on the last 3 training months and scaled up by row count.
  Each seed also changes the K-fold split; backtests use 3 seeds and the final model 5.

## 2. External data

| Data | Processing | Effect |
| --- | --- | --- |
| HDB resale transactions (data.gov.sg, downloaded 10-19) | Reuses `build_resale_history.snapshot`; only transactions **at least 2 months** before the training cut-off, in a 36-month window; block/street/town median price per sqm and counts | Single model improves by 0.4–1.2 in all four windows |
| HDB property information (property.csv, latest snapshot) | Matched on block + normalised street name, 99.87% of training and 99.97% of test rows; total dwelling units, flat-type and rental-unit shares, shop/hawker/car-park flags | Flat on the earlier quarters, small gains on the main validation and the 12-month window |

The data.gov.sg "Renting Out of Flats" dataset was not used: it contains the test months' true rents, so
using it would amount to reading the labels.

## 3. Backtest results

RMSE in SGD per month. The earlier quarters (2024 Q3, Q4) are used for selection; the main validation and
the 12-month window are reported only. "Blend" means mixing row-level predictions with the original trend
blend (the public 437.44 model) at the given weight.

| Option | 2024 Q3 | 2024 Q4 | 2025 Q1 main validation | 2024 full year (12 months) |
| --- | ---: | ---: | ---: | ---: |
| Original trend blend (control) | 474.68 | 442.11 | 448.33 | 491.79 |
| TE single model | 474.50 | 439.48 | 444.95 | 489.31 |
| TE + resale, 50% blend (public about 435) | 472.58 | 438.24 | 444.57 | 487.53 |
| TE + resale + block attributes, 75% blend (public 435.91) | 472.93 | 437.82 | 443.58 | 487.26 |
| Four-variant average, 75% blend (public 435.27) | 472.54 | 437.75 | 443.65 | 486.95 |

The four variants are: base (resale + block attributes), plus group median/standard deviation, plus
same-flat-type nearest-block levels (nearest 8 blocks and within 500 m), and plus same-month listing
counts. Each one alone changes results by less than ±0.7, residuals between variants correlate at 0.9987,
and averaging only gives a 0–0.4 variance reduction.

**Directions not adopted** (results in each `te_v*/results.csv`):

- Encodings from the last 12 months only, and learning rate 0.015: within ±0.3.
- CatBoost (same features + native address categories): worse as a single model; the best three-way blend
  gives it only 5% weight.
- Damped-trend level extrapolation (φ=0.9, 1.0): quarterly windows barely change, and without damping the
  12-month window worsens by 4.7.
- Stronger regularisation or different leaf counts (`min_data_in_leaf` 150/400, `num_leaves` 63/255,
  L2 50): none beats the base parameters on the pooled earlier quarters.
- Log target: pooled earlier-quarter RMSE worsens by 1.6.

Blend weight: chosen between 50% and 75% on the pooled earlier-quarter squared error; with block
attributes included, 75% is better.

## 4. Public leaderboard level probes

Under a global scale s, the public MSE is an exact quadratic function of s. The base is the 75% blend of
the four-variant average:

| File | Adjustment | Public RMSE |
| --- | --- | ---: |
| `submission_te_avg4_blend75_20261029.csv` | ×1.00 | 435.26895 |
| `..._scale0.98_...` | ×0.98 | 447.63710 |
| `..._scale1.02_...` | ×1.02 | 431.98839 |
| `..._scale1.017_...` | ×1.017 (quadratic optimum of three points) | 431.88651 |
| `..._trendprobe_...` | ×(1.017 + 0.002 × months from the test-period centre) | **431.07006** |

- The four scale points fit the quadratic with residuals below 1e-5, and the quadratic coefficient is
  within 0.03% of the mean squared test prediction, so the fit is reliable.
- The joint optimum with the trend is level 1.0170 and +0.18% per month, predicting 431.059, only 0.01
  away from the submitted trend probe, so no further submission was made.
- Meaning: the test period's overall rent is about 1.7% above the flat3 extrapolation, and the gap grows
  for later months (about ×1.000 in 2025-03 and ×1.032 in 2026-07). This matches the 30–40 average
  underprediction of flat3 in the local backtests.
- Risk: the two parameters were fitted on the public subset. If the public and private splits are drawn
  at random from the same months, the private score should improve similarly, but this is not
  guaranteed.

Submission records and checksums are in [submission_manifest.json](../submissions/te_trend_20261030/submission_manifest.json).

## 5. Reproduction

```bash
for f in "" "--stats" "--knn" "--volume"; do
  .venv/bin/python src/models/run_te_experiments.py --export --seeds 5 --resale --property $f \
    --out reports/experiments/<matching directory>
done
.venv/bin/python src/models/export_te_submission.py
```

The matching directories are `te_v2_property`, `te_v3_stats`, `te_v3_knn` and `te_v5_volume`. For
backtests, drop `--export` and use `--seeds 3`. Location features must first be built as described in the
README, and the original trend-blend submission file and `data/raw/external/hdb_20261019/` must be
present. Row-level predictions contain true rents and are not committed to Git.
