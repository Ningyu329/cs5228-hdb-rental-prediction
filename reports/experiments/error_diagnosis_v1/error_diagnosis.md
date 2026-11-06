# Large Error Mechanism Diagnosis

This round only analyses local validation errors of the submitted model; it trains or modifies no rent model, removes no rows and does not read the competition test set.

## Key findings

1. On the main validation, records with actual rent <=2000 are only 5.52% of rows but contribute 35.20% of SSE; 308 records are overpredicted by more than 1000 and 49 underpredicted by more than 1000. The most prominent tail problem is a small number of cheap rentals predicted at the normal market level, not only expensive flats being underpredicted.
2. 98.21% of the worst 5% of rows have historical comparables, but "comparable" does not mean the same unit or enough recent records; only 35.42% of all validation rows have at least 5 same-block same-flat-type records in the past twelve months.
3. A fixed identification diagnostic reaches ROC-AUC 0.491–0.529 across three windows and shows no stable ability to identify severe overpredictions. With history distributions and the original prediction added, only 2.24% of the top-5% risk rows on the main validation are actually overpredicted by more than 1000, against a 3.45% base rate. This does not prove labels are random; it only shows this diagnostic found no usable identification rule.
4. The main validation has 996 groups of identical normalised inputs with different rents. The finite-sample RMSE of post-hoc group means is 171.86, but this constraint is only about 14.69% of the current SSE. The current error of about 448 therefore cannot all be called irreducible noise, and 171.86 is not a true lower bound for new rows.
5. Even if the worst 5% of predictions were made exactly right after the fact, the RMSE over all rows would still be 338.03. Fixing a few outliers alone cannot be expected to reach 200, and removing those records to report a better score is not acceptable.

Identification diagnostic details are in `../overprediction_probe_v1/probe_report.md`. The original model and submitted predictions are unchanged.

## Error direction and rent level

| Window | RMSE | Overpredicted >1000 | Underpredicted >1000 | Share with actual rent <=2000 | Their SSE share |
| --- | ---: | ---: | ---: | ---: | ---: |
| q3_2024 | 474.68 | 315 | 96 | 5.7% | 33.1% |
| q4_2024 | 442.11 | 276 | 46 | 5.2% | 37.0% |
| primary_2025 | 448.33 | 308 | 49 | 5.5% | 35.2% |

Grouping by actual rent naturally shows low rents overpredicted and high rents underpredicted, even for a sound conditional-mean model. This only locates where errors concentrate and cannot justify lowering predictions for unknown low-rent test rows.

## Historical comparable coverage

Comparable conditions are town, block, street, flat type, flat model, floor area and lease commencement year, excluding unit number; the history window ends strictly before the whole validation window.

| Window | Validation rows with historical comparables | Worst 5% with historical comparables | With at least 5 recent block/flat-type records | Spearman correlation of historical dispersion and absolute error |
| --- | ---: | ---: | ---: | ---: |
| q3_2024 | 99.0% | 98.0% | 36.2% | 0.151 |
| q4_2024 | 98.8% | 98.6% | 36.2% | 0.129 |
| primary_2025 | 99.2% | 98.2% | 35.4% | 0.132 |

Historical dispersion is computed within the same block and flat type over the past twelve months, after subtracting each historical month's flat-type mean rent. Correlation is not causation and estimates are unreliable for sparse groups; any future training feature must be built from past/OOF information for every training row.

## Rent variation among identical observable inputs

| Window | Rows in repeated input groups | Input groups with conflicting rents | Full-sample RMSE of post-hoc group means | Same RMSE on raw unnormalised inputs |
| --- | ---: | ---: | ---: | ---: |
| q3_2024 | 2473 | 1040 | 180.72 | 154.61 |
| q4_2024 | 2044 | 861 | 156.99 | 133.47 |
| primary_2025 | 2408 | 996 | 171.86 | 144.67 |

The post-hoc mean of each complete input group uses the validation labels themselves and is descriptive only. It is an empirical bound for deterministic predictions given these inputs and this finite sample, not a deployable prediction or an estimate of irreducible error for new rows. Singleton groups have zero error here, which does not mean they are predictable.

The lower post-hoc RMSE on raw formats does not by itself show that keeping the original casing would improve the model. Finer inputs add post-hoc fitting freedom; even a random unique id would drive this value to zero. Whether they carry generalisable information must be checked with temporal validation.

## Impact of tail errors

| Window | Worst 5% SSE share | Overall RMSE if that 5% had zero error | Worst 10% SSE share | Overall RMSE if that 10% had zero error |
| --- | ---: | ---: | ---: | ---: |
| q3_2024 | 41.5% | 363.15 | 57.5% | 309.36 |
| q4_2024 | 45.3% | 326.99 | 61.0% | 276.14 |
| primary_2025 | 43.2% | 338.03 | 59.1% | 286.64 |

The table keeps all rows in the denominator; it is not a score after removing hard rows. It relies on errors known after the fact and is not an achievable improvement.

## Cleaning and raw-format review

- All normalised main input columns were recomputed row by row against the raw training table; labels and row order match.
- FURNISHED and FEE have only 1 and 1 distinct values and carry no information between rows.
- 35331 rows use the space-separated flat type, only from 2021-01 to 2021-12; no 2024/2025 validation row uses it. Raw flat-type mean differences are confounded with year and flat-type mix, so means alone cannot show that merging formats hurts the model.

## Usage limits and next steps

- Actual-rent bins, tail rows and conflicting groups are label-dependent diagnostics and cannot be used to route test rows, drop rows or flag cheap flats.
- Historical coverage, past rent distributions and neighbourhoods are available at prediction time; any gain must be verified with temporal backtests, not assumed from correlation.
- Missing unit number, actual floor, renovation or contract terms may explain part of the within-group variation, but these data cannot verify the cause, so not all residuals can be called noise.
- First evaluate whether past rent distributions or neighbour features separate rows within the same location, and check whether large errors concentrate on historically cheap rentals or specific observable attributes. Do not drop low-rent records without confirmation.
- All validation windows have been inspected several times, so this report is not an independent final test.

## Verification and artefacts

- 65 synthetic-data tests pass, covering history-window boundaries, query labels never entering features, input-conflict statistics, group SSE conservation and the fixed alert budget.
- An independent script checks 25,984 validation records and 36 groupings, and recomputes the input-conflict SSE a different way; the six classification diagnostic scores reproduce exactly after fresh training.
- The SHA-256 of the submitted CSV was rechecked; the original predictions were not modified.
- `error_mechanisms*.csv` and `diagnosis.json` are aggregate diagnostics; `tail_samples_*.csv` contain private row-level data, are kept locally only and are not committed to Git.
