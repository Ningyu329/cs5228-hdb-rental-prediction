# Location Feature Ridge Ablation

## Design

- temporal: train 2021-01 to 2024-12 (141,076 rows), validate 2025-01 to 2025-03 (8,924 rows).
- temporal_backtest_1: train 2021-01 to 2024-09 (132,802 rows), validate 2024-10 to 2024-12 (8,274 rows).
- temporal_backtest_2: train 2021-01 to 2024-06 (124,016 rows), validate 2024-07 to 2024-09 (8,786 rows).
- random: train 2021-01 to 2025-03 (120,000 rows), validate 2021-01 to 2025-03 (30,000 rows).

All variants use the same Ridge parameters and fixed splits; imputer, scaler and one-hot encoder are fitted on each training fold only.

## Main temporal validation

| Feature set | Fields | RMSE | Improvement over this run's main-table baseline | Improvement % |
| --- | ---: | ---: | ---: | ---: |
| Main-table Ridge | 11 | 562.6666 | +0.0000 | +0.00% |
| + HDB coordinates and block attributes | 23 | 562.4358 | +0.2309 | +0.04% |
| + HDB + MRT | 26 | 562.0663 | +0.6004 | +0.11% |
| + HDB + MRT + schools | 29 | 562.0281 | +0.6385 | +0.11% |
| + HDB + MRT + schools + malls | 32 | 562.0597 | +0.6069 | +0.11% |
| + HDB + schools + malls (no MRT) | 29 | 562.5459 | +0.1207 | +0.02% |

## Earlier windows and random split

| Feature set | Split | RMSE | Improvement over same-split baseline |
| --- | --- | ---: | ---: |
| Main-table Ridge | temporal_backtest_1 | 529.8556 | +0.0000 |
| + HDB coordinates and block attributes | temporal_backtest_1 | 529.4256 | +0.4300 |
| + HDB + MRT | temporal_backtest_1 | 529.0222 | +0.8334 |
| + HDB + MRT + schools | temporal_backtest_1 | 528.9965 | +0.8590 |
| + HDB + MRT + schools + malls | temporal_backtest_1 | 528.9253 | +0.9303 |
| + HDB + schools + malls (no MRT) | temporal_backtest_1 | 528.9210 | +0.9346 |
| Main-table Ridge | temporal_backtest_2 | 546.6978 | +0.0000 |
| + HDB coordinates and block attributes | temporal_backtest_2 | 545.8737 | +0.8241 |
| + HDB + MRT | temporal_backtest_2 | 545.7541 | +0.9437 |
| + HDB + MRT + schools | temporal_backtest_2 | 545.7136 | +0.9842 |
| + HDB + MRT + schools + malls | temporal_backtest_2 | 545.9325 | +0.7653 |
| + HDB + schools + malls (no MRT) | temporal_backtest_2 | 545.9508 | +0.7470 |
| Main-table Ridge | random | 505.1789 | +0.0000 |
| + HDB coordinates and block attributes | random | 504.8509 | +0.3280 |
| + HDB + MRT | random | 504.7496 | +0.4293 |
| + HDB + MRT + schools | random | 504.7212 | +0.4578 |
| + HDB + MRT + schools + malls | random | 504.6980 | +0.4809 |
| + HDB + schools + malls (no MRT) | random | 504.7271 | +0.4518 |

## Interpretation and limitations

- The previously recorded Ridge temporal RMSE is 562.5888; this rerun with the same configuration gives 562.6666. Environments differ, so all improvements use this same-environment rerun as reference.
- Lowest main temporal validation score: hdb_mrt_school; lowest without MRT: hdb.
- The best main-validation variant lowers RMSE by 0.6385 (0.11%) against this baseline, a limited gain. It beats the matching baseline in 2 of 2 earlier windows.
- Mean prediction minus actual rent on the main validation: baseline +316.44, lowest-RMSE variant +316.85. Positive values mean overall overprediction, which location features do not fix.
- The lowest score only selects candidates for later experiments; it does not replace independent test performance or show statistical significance. Earlier windows check stability; the random split is auxiliary.
- MRT temporal mode: snapshot. With snapshot or unknown, MRT results are exploratory because currently open stations may not have been open in early rental months.
- School, mall and some block attributes are also static snapshots, so removing MRT does not prove the other auxiliary fields are historically available.
- Validation windows are 3 months by default while the test set extends further, so long-horizon extrapolation error may be larger.
- ADDRESS_KEY and POSTAL_CODE are not used as model inputs. Constant flags such as address-match indicators stay in the preset variants; Ridge regularisation and scaling match the baseline.
- No hyperparameters were tuned, baseline_config.json was not modified, and no Kaggle submission was generated.

Group errors are in group_errors.csv: RMSE, MAE and signed bias by town, flat type, month and region for the baseline and main candidates.

## Next experiments

First compare models or time features that capture nonlinear time changes, and check whether overall overprediction relates to long-term linear trend extrapolation. This is only a hypothesis and cannot be inferred from mean bias alone. Keep the same windows and keep MRT as a separate control with its availability caveat.
