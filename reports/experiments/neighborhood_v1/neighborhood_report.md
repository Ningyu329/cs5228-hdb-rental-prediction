# Unit-Area Rent and Historical Spatial Features

## Conclusion

This round compares 7 new recipes and their blends with the current model and finds no improvement that meets the adoption threshold. The best new combination lowers the pooled RMSE over the two historical quarters from 459.1748 to 458.8237, only 0.3511, and on the early-2025 validation it rises from 448.3346 to 448.8767. The current model is therefore kept and no new Kaggle submission is generated or uploaded. The existing public score remains 437.44368.

"Current model" here means the submitted CatBoost, LightGBM and damped-trend blend, not the original Ridge baseline.

## Method and time boundaries

The new recipes use the same main data and housing/location fields and test the following changes.

1. **Unit-area rent.** The target becomes monthly rent divided by floor area, and predictions are multiplied back by area. Training and early stopping use area-squared weights normalised to a mean weight of 1. For given validation rows, the weighted squared error in unit-area terms differs from the currency squared error only by a constant factor, so the change of target does not drift away from currency RMSE. Tree structure and regularisation can still give different results.
2. **Historical rent distributions.** For groups with the same flat attributes, and for same-block same-flat-type groups, the count, mean, median, standard deviation and 10th/90th percentiles of unit-area rent over the past 12 months are computed. "Same attributes" does not mean the same flat.
3. **Spatial neighbours.** Historical rentals are first aggregated into the mean unit-area rent of each block and flat type, then the nearest 16 and 64 blocks with the same flat type are found. Block count, mean, standard deviation, median rent and distance are extracted. Blocks are weighted equally so that one block's repeated rentals do not crowd out its neighbours.
4. **Combination and capacity.** The two feature groups are combined to predict either rent or unit-area rent; a LightGBM with more leaves and a strongly regularised depth-5 CatBoost are also tried.

Historical statistics for training rows exclude the whole current month; the validation window only uses the history snapshot at the training cut-off and is never rolled forward with validation-period rents. Category vocabularies are built from training data only.

Each outer window chooses boosting rounds on the last 3 training months and then refits on all outer training data. Outer validation targets are never used for feature construction, encoding or early stopping. Retraining the original LightGBM on both quarters reproduces the old predictions row for row.

The ideas draw on earlier cohorts' [spatial neighbour and distribution features](https://github.com/Concyclics/NUS-CS5228-project/blob/main/detail.md) and [shallow CatBoost case](https://github.com/cs5228-group-1/cs5228-final-project). Data, validation and leaderboards from other years are not directly comparable. This round uses its own past-information implementation. Weighted validation parameters follow the [LightGBM API](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMRegressor.html).

## Selection rule

The full recipes were written to `experiment_manifest.json` before any new validation result was read. Each new model is blended with the current model at 25%, 50% and 100% weight; with the current model itself, 22 options are compared.

Selection uses only the 17,060 validation records of 2024 Q3 and Q4. A new option is adopted only if the pooled RMSE falls by at least 1.0 and neither quarter worsens by more than 1.0. This threshold is the practical adoption rule for this round, not a significance test.

Only after the choice was written to `selection_locked.json` were the best new challenger's early-2025 and 6-month-window results checked. These windows were viewed in earlier rounds and are not fresh hold-out sets; the 6-month window also overlaps the quarterly records, with a different training cut-off.

## Results

Each row shows the lowest pooled RMSE among the three preset blend weights for that recipe. The difference is new option minus current model; only negative values are improvements.

| Recipe | New model weight | Two-quarter pooled RMSE | Difference from current model |
| --- | ---: | ---: | ---: |
| Current model | — | 459.1748 | 0.0000 |
| Unit-area target only | 25% | 459.3719 | +0.1971 |
| Historical distributions | 25% | 460.1229 | +0.9481 |
| Spatial neighbours | 25% | 460.0638 | +0.8890 |
| Historical distributions plus spatial neighbours | 25% | 460.2104 | +1.0356 |
| Unit-area target plus both feature groups | 25% | 458.8237 | −0.3511 |
| Both feature groups plus more flexible LightGBM | 25% | 460.5094 | +1.3346 |
| Shallow CatBoost | 25% | 458.8458 | −0.3290 |

The best new challenger is "75% current model + 25% unit-area target model with both new feature groups". Its results by window:

| Validation window | Rows | Current model RMSE | New challenger RMSE | Difference |
| --- | ---: | ---: | ---: | ---: |
| 2024 Q3 | 8,786 | 474.6845 | 473.6973 | −0.9872 |
| 2024 Q4 | 8,274 | 442.1101 | 442.4826 | +0.3725 |
| Two quarters pooled | 17,060 | 459.1748 | 458.8237 | −0.3511 |
| 2025 Q1 | 8,924 | 448.3346 | 448.8767 | +0.5421 |
| Second half of 2024, single forecast | 17,060 | 460.0744 | 459.5564 | −0.5181 |

On the early-2025 validation the MAE also rises, from 326.4557 to 327.5352. The new challenger does not meet the historical selection threshold and shows no consistent advantage in later diagnostics.

## New understanding of the error

Over the past 12 months about 90% of validation records have same-block same-flat-type history, but only about 35–37% have at least 5 such records, so quantile and volatility estimates can still be unstable. The median distances to the 16th and 64th nearest blocks are about 0.36 and 0.95 km, so the spatial queries do cover local areas rather than city-wide averages.

That the model uses new features does not mean validation improves. In the unit-area combined model, spatial features account for about 73–74% of training split gain, yet neither the single model nor the blends show a stable enough advantage. This does not prove spatial information is useless; it only shows this round's representation adds no stable extra gain.

For early-2025 rows with actual rent at most 2,000, RMSE falls slightly from 1,131.74 to 1,127.43, while for the rest it rises from 371.30 to 372.75. Cheap rentals are still widely overpredicted, so the new recipes do not solve the large-error problem found earlier. This grouping is post-hoc analysis only; models cannot be chosen, nor validation rows removed, based on unknown true rent.

## Next priorities

1. **Multi-origin validation matched to the test horizon.** Selection so far mainly compares quarterly forecasts, while the test set extends to July 2026. The next round should fix several earlier training cut-offs in advance, compare 6–12-month or longer forecasts, check bias month by month, and then choose the trend and training window. Earlier 6- and 12-month results were mainly diagnostic and were not used for selection in this round.
2. **Location representations other than neighbour rents.** Cheap comparisons include the identity of the nearest MRT station and mall, geographic cluster labels, and region by flat-type interactions. Historical availability must be confirmed first; stations opening in the future cannot be treated as open at the time. These are directions to test, with no promised gain.
3. **Check the rental definition and linkable information.** If a large improvement is the goal, first find out whether observable information explains why some flats with the same attributes rent cheaply. Only data allowed by the competition and legitimately linkable may be used; test rents must never be backfilled. Current results neither support a promise of reaching 200 nor prove that 200 is unreachable.

This round does not support expanding similar tuning grids to chase gains of a few tenths; more flexible models did not improve these validations either, but that does not mean every model has reached its limit.

## Paths and reproduction

The input is `data/processed/train_location_features.csv`; the test set is not read. Model and feature caches are stored in `models/neighborhood_cache/` and `models/neighborhood_features/` without overwriting existing models. Row-level predictions are kept locally only; the CSVs allowed into Git are reviewed summaries, selections, feature importance and group diagnostics.

```bash
.venv/bin/python src/models/run_neighborhood_experiments.py --threads 3 --output-dir reports/experiments/neighborhood_reproduction
.venv/bin/python src/models/inspect_neighborhood_experiments.py --output-dir reports/experiments/neighborhood_reproduction
.venv/bin/python -m unittest discover -s tests
```

Use a new output directory after changing source, dependencies or parameters. 72 tests pass; the independent check recomputes 48 metrics and 22 selection options, reloads 18 models to verify predictions, and rebuilds validation features directly without the feature cache. The SHA256 of the original submission file is unchanged.
