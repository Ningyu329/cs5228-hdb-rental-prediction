# CS5228 HDB Monthly Rent Prediction

Team project for the CS5228-2610 Kaggle competition (**CS5228 Avengers, Team 31**):
Wang Ningyu, Xie Lin, Zhang Baowen, Zhu Tianyi.

The task is to predict the monthly rent (`MONTHLY_RENT`) of Singapore HDB flats from flat attributes,
location and time. Submissions are scored by root mean squared error (RMSE) in SGD per month.

## Final result

| Submission | Main local validation RMSE (Jan–Mar 2025) | Kaggle public RMSE |
| --- | ---: | ---: |
| Ridge baseline (main table only) | 562.67 | 633.17 |
| Tuned CatBoost + LightGBM trend blend | 448.33 | 437.44 |
| Final model: target-encoding LightGBM, 75/25 blend with the trend blend | **443.65** | 435.27 |
| Final model + market-level calibration (**final submission**) | — | **431.07** |

The final submission is `submission_te_avg4_blend75_trendprobe_20261029.csv`. Its checksum and every
leaderboard probe are recorded in
[reports/submissions/te_trend_20261030/submission_manifest.json](reports/submissions/te_trend_20261030/submission_manifest.json).

Local and public scores use different rows and months, so they are not directly comparable.

## Approach

1. **Cleaning** ([src/data/clean_data.py](src/data/clean_data.py)): normalises text, flat types, dates
   and numbers without dropping rows. Identical inputs with different rents are kept because they are
   not necessarily the same flat.
2. **Location features** ([src/features/build_location_features.py](src/features/build_location_features.py)):
   exact `TOWN + BLOCK + STREET` matching to the HDB block table (100% coverage), coordinates, planning
   area, building age, and Haversine distances and counts for MRT stations, schools and malls.
3. **Housing and time features** ([src/features/build_stage5_features.py](src/features/build_stage5_features.py),
   [src/features/build_temporal_features.py](src/features/build_temporal_features.py)).
4. **Trend blend** ([src/models/run_rmse_improvement.py](src/models/run_rmse_improvement.py)): tuned
   CatBoost and LightGBM plus a LightGBM model on rent with the monthly market level removed. This was
   the previous best model and is 25% of the final blend.
5. **Out-of-fold target encoding** ([src/models/run_te_experiments.py](src/models/run_te_experiments.py)):
   - Rent is divided by a mix-adjusted monthly market level.
   - Five-fold out-of-fold smoothed means are computed for block × flat type,
     block × flat type × floor area, block, street × flat type, subzone × flat type and
     town × flat type × flat model. Query rows are encoded from the whole training window and never
     see their own labels.
   - Two public data.gov.sg tables are added: block-level resale price per square metre, using only
     transactions at least two months before the cut-off, and HDB block composition (unit mix,
     rental-unit share, facilities).
   - LightGBM predicts the level-adjusted rent. Future months use the last three months' level.
   - Four feature variants, each averaged over five seeds, are averaged together and blended 75/25
     with the trend blend. The weight is chosen on the 2024 Q3/Q4 backtests.
6. **Market-level calibration** ([src/models/export_te_submission.py](src/models/export_te_submission.py)):
   - The public MSE is an exact quadratic function of a global prediction scale, so four
     leaderboard probes identify the best level and monthly trend: about +1.7% overall and +0.18% per
     month.
   - These two parameters are the only quantities fitted to the public leaderboard. Every model and
     blending choice uses local temporal validation.

The HDB "Renting Out of Flats" dataset on data.gov.sg was deliberately **not** used, because it
contains the test-period rents themselves.

### Validation

The test period (March 2025 – July 2026) lies after the training period (January 2021 – March 2025),
so all model selection uses forward temporal splits:

- **Selection windows**: 2024 Q3 and 2024 Q4, each trained only on earlier months; squared errors
  are pooled.
- **Main validation**: January–March 2025, used for reporting rather than selection.
- **Long-horizon check**: train to 2023-12, validate on all of 2024 (12 months ahead).
- Boosting rounds are chosen on an inner split of the last three training months.

| Window | Trend blend | Final 75/25 blend |
| --- | ---: | ---: |
| 2024 Q3 | 474.68 | 472.54 |
| 2024 Q4 | 442.11 | 437.75 |
| 2025 Q1 (main) | 448.33 | 443.65 |
| 2024 full year (12-month horizon) | 491.79 | 486.95 |

The target-encoding model is the only change in the project that improved every window by more than
one point.

## Repository layout

```text
src/
  data/       cleaning, data audit, public HDB download
  eda/        exploratory figures and error diagnostics
  features/   location, housing, temporal, resale and neighbourhood features
  models/     baselines, tree models, experiments, submission export
tests/        unit tests on synthetic data (python -m unittest)
reports/
  experiments/   per-experiment summaries (results.csv, selection, reports)
  submissions/   submission manifests, checksums and leaderboard records
  figures/       EDA figures used in the report
  progress_report/  mid-term slides
requirements-*.txt   pinned dependencies
```

Private competition data, processed features, row-level predictions, trained models and submission
CSVs are excluded by `.gitignore` and must be rebuilt locally.

## Setup

Python 3.12 is used.

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements-stage5.txt
```

Optional extras: `requirements-xgboost.txt`, `requirements-keras.txt` and `requirements-tabm.txt`, only
for the corresponding experiments.

Place the competition files under `data/raw/`:

```text
data/raw/train.csv
data/raw/test.csv
data/raw/auxiliary/   (sg-hdb-block.csv, sg-mrt-stations.csv, sg-schools.csv,
                       sg-shopping-malls.csv, sg-coe-prices.csv, sg-stock-prices.csv)
```

Download the two public HDB tables, resale prices and property information, into
`data/raw/external/hdb_20261019/`:

```bash
.venv/bin/python src/data/download_public_hdb.py
```

## Reproducing the final submission

Run from the repository root. Steps 4–5 train many models and take a few hours on a laptop CPU.

```bash
# 1. Clean data and build features
.venv/bin/python src/data/clean_data.py
.venv/bin/python src/features/build_location_features.py

# 2. Baselines, model comparison and tuning (the trend blend builds on these)
.venv/bin/python src/models/train_baseline.py
.venv/bin/python src/models/run_stage5.py
.venv/bin/python src/models/run_tree_tuning.py --output-dir reports/experiments/tree_tuning_local

# 3. Trend-blend experiment and its full-data submission
.venv/bin/python src/models/run_rmse_improvement.py --output-dir reports/experiments/rmse_improvement_local
.venv/bin/python src/models/export_rmse_submission.py

# 4. Target-encoding variants: backtests (omit --export) and full-data test predictions
.venv/bin/python src/models/run_te_experiments.py --export --seeds 5 --resale --property --out reports/experiments/te_v2_property
.venv/bin/python src/models/run_te_experiments.py --export --seeds 5 --resale --property --stats --out reports/experiments/te_v3_stats
.venv/bin/python src/models/run_te_experiments.py --export --seeds 5 --resale --property --knn --out reports/experiments/te_v3_knn
.venv/bin/python src/models/run_te_experiments.py --export --seeds 5 --resale --property --volume --out reports/experiments/te_v5_volume

# 5. Average, blend and calibrate
.venv/bin/python src/models/export_te_submission.py
```

Backtest mode (without `--export`) compares against the trend blend's row-level validation
predictions in the `incumbent` column of `reports/experiments/keras_v1/predictions_*.csv`. Those files
are written by `src/models/run_keras_experiments.py` and kept local only.

`export_te_submission.py` reproduces the submitted file from the saved predictions (verified to within
1e-12). Use `--level 1 --trend 0` to get the uncalibrated blend.

**Source fingerprints.** The experiment manifests record SHA-256 hashes of the source files at run
time. `export_rmse_submission.py` refuses to run if those files changed. A later commit translated
the comments and report strings in the code to English. An AST comparison confirms that only string
literals changed, not logic. To regenerate the trend blend with matching fingerprints, check out the
pre-translation commit `05e8ddf` for step 3.

## Experiment index

Every experiment keeps its own directory with summary metrics; row-level outputs stay local.
Except `te_report.md`, the per-experiment write-ups listed below are in Chinese; their tables and
`results.csv` files are language-neutral.

| Experiment | Entry point | Results | Outcome |
| --- | --- | --- | --- |
| Data audit | `src/data/audit_cleaning.py` | [report](reports/data_quality_after_cleaning.md) | 100% address match, no rows dropped |
| Location ablation (Ridge) | `src/models/run_location_ablation.py` | [location_ablation_v1](reports/experiments/location_ablation_v1/) | 562.67 → 562.03 |
| Model and feature comparison | `src/models/run_stage5.py` | [stage5_v1](reports/experiments/stage5_v1/) | Tree models ≈ 452 |
| CatBoost / LightGBM tuning | `src/models/run_tree_tuning.py` | [tree_tuning_v1](reports/experiments/tree_tuning_v1/) | 450.47 |
| Rounds, recency, history, trend | `src/models/run_rmse_improvement.py` | [rmse_improvement_v1](reports/experiments/rmse_improvement_v1/) | 448.33, public 437.44 |
| Segment correction, macro | `src/models/run_segment_experiments.py` | [segment_v1](reports/experiments/segment_v1/) | No stable gain |
| Neighbourhood, unit-area target | `src/models/run_neighborhood_experiments.py` | [neighborhood_v1](reports/experiments/neighborhood_v1/) | No stable gain |
| XGBoost | `src/models/run_xgboost_experiments.py` | [xgboost_native_v1](reports/experiments/xgboost_native_v1/) | No stable gain |
| Raw views, resale history, hierarchical model | `src/models/run_data_rethink.py` | [data_rethink_v3](reports/experiments/data_rethink_v3/) | Public 437.46 |
| Error diagnosis | `src/eda/analyze_error_mechanisms.py`, `src/eda/probe_large_overprediction.py` | [error_diagnosis_v1](reports/experiments/error_diagnosis_v1/), [overprediction_probe_v1](reports/experiments/overprediction_probe_v1/) | Residual floor from unobserved unit differences |
| Keras embedding networks | `src/models/run_keras_experiments.py` | [keras_v1](reports/experiments/keras_v1/) | No stable gain |
| TabM | `src/models/run_tabm_experiments.py` | [tabm_v1](reports/experiments/tabm_v1/) | No stable gain |
| **Target encoding + external data** | `src/models/run_te_experiments.py` | [te_report.md](reports/experiments/te_report.md), `te_v*/` | **443.65, public 435.27 / 431.07** |

`inspect_*.py` scripts independently reload saved models and recompute metrics for the matching
experiments.

## Main limitations

- Rows with identical observable inputs still differ by several hundred SGD; floor, unit condition
  and contract terms are not in the data. Different models' residuals correlate at about 0.98–0.999.
- Static auxiliary tables (MRT, schools, malls, HDB property) are snapshots, not historical versions.
- The level calibration was fitted on the public leaderboard subset and assumes the private split
  covers the same months.

## Tests

```bash
.venv/bin/python -m unittest discover -s tests
```

103 tests run on synthetic data and do not need the competition files.
