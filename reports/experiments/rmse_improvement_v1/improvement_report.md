# Sequential Experiments to Reduce RMSE

## Rules

- The control is a same-environment rerun of the equal-weight 800-round CatBoost / tuned LightGBM blend; the historical score 450.4735 is reference only.
- Each stage selects only on pooled squared error over 2024 Q3 and Q4, keeping the previous option as a candidate. Later stages depend on earlier choices; this is bounded sequential tuning, not an exhaustive search.
- Recency options and history statistics are screened with LightGBM first, then one new option is confirmed with CatBoost to limit compute.
- CatBoost early-stops on an earlier inner three-month validation; the 800/1600/2400 caps share one 2400-round trajectory and are refitted with the best rounds below each cap.
- Rent statistics for training rows use earlier months only; the whole validation query is frozen at the training cut-off and never updated with same-month or future validation rents.
- Trend experiments remove the overall monthly rent level and fit residuals on static housing features; future levels use the last-three-month constant or a six-month damped trend from the last 12/24 months.
- 2025 Q1 and the 6/12-month windows are reported only after selection. The main validation has been viewed before; long windows overlap the tuning quarters and are diagnostics from different origins.
- The competition test set is not read, no submission is generated, and duplicates or extreme rents are not removed.

## Stage selection

| Stage | Selected option | Earlier-quarter pooled RMSE |
| --- | --- | ---: |
| 01_caps | cap_1600_blend | 460.1692 |
| 02_recency | cap_1600_blend | 460.1692 |
| 03_history | cap_1600_blend | 460.1692 |
| 04_trend | trend_damped24_weight25 | 459.1748 |

## Main temporal validation: January-March 2025

| Option | RMSE | MAE | Mean bias |
| --- | ---: | ---: | ---: |
| control_800_blend | 450.4735 | 331.1260 | -37.38 |
| cap_1600_blend | 449.3854 | 330.0369 | -36.40 |
| trend_damped24_weight25 | 448.3346 | 326.4557 | -20.73 |

Local control RMSE 450.4735, pre-locked option 448.3346; reduction +2.1389 (+0.47%). A negative value means the new option is worse on the main validation; the earlier choice is still kept and not reselected on the main validation.

## Long-horizon diagnostics

| Window | Option | RMSE | MAE | Bias |
| --- | --- | ---: | ---: | ---: |
| long_6m | control_800_blend | 462.3438 | 341.8063 | -51.55 |
| long_6m | trend_damped24_weight25 | 460.0744 | 333.7802 | -12.01 |
| long_12m | control_800_blend | 492.2118 | 370.9976 | -45.35 |
| long_12m | trend_damped24_weight25 | 491.7916 | 358.2847 | +39.65 |

## Limitations and artefacts

- Rent statistics do not identify individual flats; the same block and flat type can still have different rents. Rows without history at the start of a window stay empty and are never filled with future means.
- Monthly mean rent can shift with flat-type and location mix, so the trend decomposition is not an independent market price index.
- Only preset windows, weights and blend ratios were tried; not every combination was tested and leaderboard performance is not guaranteed. Static HDB information still has historical-availability limits.
- `results.csv` and `selection_*.csv` hold summary scores; `stage_decisions.json` and `selection_locked.json` record the selections.
- `predictions_*.csv`, `group_errors_*.csv` and `models/rmse_improvement_cache/` are local diagnostics; row-level rents, predictions and models are not committed to Git.
- Configuration, input and implementation SHA-256 values and dependency versions are in `experiment_manifest.json`; interrupted runs with the same configuration resume.
