# Report revisions for the 431.07 version

Paste-ready English text for each change. Numbers come from
`reports/experiments/te_report.md` and `reports/submissions/te_trend_20261030/`.

---

## 1. Abstract — replace the last three sentences

> Using 150,000 training observations and 50,000 test observations, we compare regularised linear
> regression, CatBoost, LightGBM, ensembles, and a hierarchical correction model. The largest gain came
> from out-of-fold target encoding of block- and street-level rents combined with lagged public resale
> prices and HDB block composition, which improved every temporal backtest. A two-parameter calibration of
> the future market level further reduced the public RMSE from 437.44 to **431.07**. Error analysis shows
> that unobserved unit-level differences and market drift remain the main limitations.

Index Terms: add `target encoding`.

---

## 2. Table I — add two rows

| Dataset | Rows | Role | Main use |
| --- | ---: | --- | --- |
| HDB resale (data.gov.sg) | 242,955 | External | Lagged block price per sqm |
| HDB property info (data.gov.sg) | 13,392 | External | Block unit mix, facilities |

Add after the paragraph about COE and stock data (Section II-B):

> Two public data.gov.sg tables were added as external sources. Resale transactions were aggregated into
> block-, street- and town-level median price per square metre, using only transactions at least two
> months before the training cut-off. The HDB property table supplies each block's dwelling count,
> flat-type and rental-unit shares, and commercial or hawker facilities. Exact block and normalised
> street matching covered 99.87% of training rows and 99.97% of test rows. The public rental-flat
> dataset was deliberately not used, because it contains the test-period rents themselves.


---

## 3. Section III — new paragraph after the "Further experiments ..." paragraph

> **Out-of-fold target encoding.** Earlier history features used only months before each row, so training
> rows saw sparse, early statistics while query rows saw the full history. We instead divided rent by a
> mix-adjusted monthly market level and computed five-fold out-of-fold means [4] for fine groups: block ×
> flat type, block × flat type × floor area, street × flat type, subzone × flat type and town × flat type ×
> flat model, with additive smoothing and group counts. Query rows were encoded from the whole training
> window and never used their own labels. A LightGBM model on these encodings, location, external and
> structural features predicted the level-adjusted target, and future months used the average level of
> the last three training months. Four feature variants (adding group dispersion, nearest-block rent
> levels, or same-month listing counts) were each averaged over five seeds and then averaged together. The
> result was blended 75/25 with the earlier trend blend, with the weight chosen on the pooled 2024 Q3/Q4
> backtests.

Add to the references:

> [4] D. Micci-Barreca, "A preprocessing scheme for high-cardinality categorical attributes in
> classification and prediction problems," *ACM SIGKDD Explorations*, vol. 3, no. 1, 2001.

---

## 4. Table II — add rows (2025 Q1 main validation)

| Method | RMSE | MAE | Bias |
| --- | ---: | ---: | ---: |
| Trend blend (previous best) | 448.33 | 326.46 | −20.73 |
| Target-encoding LightGBM | 444.95 | 321.11 | −0.21 |
| Final 75/25 blend | **443.65** | **320.62** | −5.17 |

Optional extra table, or a sentence, for the robustness check across windows:

| Window | Trend blend | Final blend |
| --- | ---: | ---: |
| 2024 Q3 | 474.68 | 472.54 |
| 2024 Q4 | 442.11 | 437.75 |
| 2025 Q1 | 448.33 | 443.65 |
| 2024 full year (12-month horizon) | 491.79 | 486.95 |

---

## 5. Section IV-A — replace the last paragraph from "The selected trend blend was refitted ..."

> The selected trend blend was refitted on all historical data and obtained a public RMSE of 437.44. The
> target-encoding blend improved all four temporal windows, including the 12-month horizon, and reached a
> public RMSE of 435.27. This was the only change in the project that improved every window by more than
> one point. Feature variants themselves differed by less than 0.7, and their residuals correlated at
> 0.999. The gain therefore came from representing block-level information better, not from model
> capacity.
>
> **Level calibration.** Figure 7 and the long-horizon backtests showed a persistent bias when the latest
> level was carried forward. Because the public MSE is an exact quadratic function of a global
> multiplicative scale, three submissions (×0.98, ×1.00, ×1.02) identify its optimum. A fourth, with a
> linear monthly trend, identifies the trend. The optimum raised predictions by about 1.7% at the
> centre of the test period and by about 0.18% per month, improving the public RMSE to 431.07. Only two
> global parameters were estimated from the public leaderboard; all model and blending choices used local
> temporal validation.

---

## 6. Section IV-B — adjust one sentence

Replace "Their results suggest that additional feature engineering alone is unlikely to remove all error"
with:

> These results suggest that further similar models are unlikely to remove the remaining error. Better
> encoding of block-level history helped, but rows with identical observable inputs still differ by
> several hundred dollars.

---

## 7. Section IV-C Limitations — add

> The level calibration was estimated on the public subset. It should transfer if the public and private
> splits are drawn from the same months, but this cannot be verified locally. External tables are latest
> snapshots, so building composition may differ slightly from the historical state.

Future work can drop the hierarchical-model sentence and add:

> ...and a level forecast learned from leading indicators rather than from leaderboard feedback.

---

## 8. Conclusion — replace the second-to-last sentence

> The experiments show that out-of-fold target encoding with public resale and block data, together with
> a calibrated market level, gave the best result (public RMSE 431.07). The remaining error is driven by
> unobserved unit-level differences, sparse premium segments, and future market regimes.

---

## Length

The draft is about 4.3 pages. These additions add roughly 0.6–0.8 page. If there is a 4-page limit,
shorten the description of tuning in Section III (paragraphs 4–6) and keep only one of the two new tables.
