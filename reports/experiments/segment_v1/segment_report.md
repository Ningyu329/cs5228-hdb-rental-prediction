# Group Diagnostics and Targeted Optimisation

## Conclusion and model selection

Locked option: `bias_town_flat_s500`. No new submission was generated and the competition test set was not read.

This round found no meaningful stable gain: the pooled earlier-quarter RMSE only fell from 459.1748 to 459.0770 (about 0.10),
2025 Q1 moved from 448.3346 to 448.3390, and the six-month window from 460.0744 to 459.9859.
This round is therefore not claimed to beat the submitted model, and the submission is not replaced. The original model's Kaggle public score is 437.44368;
the group-correction option has no Kaggle score. The lock keeps the option selected on earlier windows and is not reselected on the later main validation.

| Stage | Locked option | Earlier-quarter pooled RMSE |
| --- | --- | ---: |
| 01_bias | bias_town_flat_s500 | 459.0770 |
| 02_trend | bias_town_flat_s500 | 459.0770 |
| 03_experts | bias_town_flat_s500 | 459.0770 |
| 04_macro | bias_town_flat_s500 | 459.0770 |

Best pooled earlier-quarter RMSE among the new options of each stage:

| Direction | Best new option | Pooled RMSE | Interpretation |
| --- | --- | ---: | --- |
| Current global control | locked_448_control | 459.1748 | Model locked in the previous round |
| Town and flat-type bias correction | bias_town_flat_s500 | 459.0770 | Tiny gain; one quarter better, one worse |
| Group residual trend | trend_town_flat_s500 | 459.5463 | Not better than the control or the corrector |
| Flat-type expert blend | expert_weight25 | 459.6935 | 25% blend with the bias corrector; no gain |
| Macro model blend | market_frozen2_weight25 | 460.1288 | The same-period information version did not help either |

The best blend with the same-period COE+stock model is 460.1760, and COE only 460.1978.
This only shows that these aggregated features, this LightGBM configuration and these blend ratios did not help; it does not show that all macro features are useless.

| Window | Option | RMSE | MAE | Mean bias |
| --- | --- | ---: | ---: | ---: |
| q3_2024 | locked_448_control | 474.6845 | 347.3841 | -12.14 |
| q3_2024 | bias_town_flat_s500 | 474.2784 | 349.1387 | -27.31 |
| q4_2024 | locked_448_control | 442.1101 | 315.2021 | -2.01 |
| q4_2024 | bias_town_flat_s500 | 442.3634 | 316.8084 | -10.67 |
| primary_2025 | locked_448_control | 448.3346 | 326.4557 | -20.73 |
| primary_2025 | bias_town_flat_s500 | 448.3390 | 326.0813 | -19.99 |
| long_6m | locked_448_control | 460.0744 | 333.7802 | -12.01 |
| long_6m | bias_town_flat_s500 | 459.9859 | 336.0331 | -27.41 |

## Priority groups: earlier backtests only

The denominator of the squared-error share is the SSE of all rows in the window. Shares sum to 100% within one grouping; different groupings overlap and cannot be added.
Bias is prediction minus actual; positive means overprediction. The squared contribution of group mean bias is a descriptive decomposition, not an achievable gain.

| Window | Group | Rows | RMSE | Bias | Share of total SSE |
| --- | --- | ---: | ---: | ---: | ---: |
| q3_2024 | jurong west / 5-room | 235 | 663.30 | -223.52 | 5.2% |
| q3_2024 | jurong west / 4-room | 216 | 621.11 | -93.49 | 4.2% |
| q3_2024 | queenstown / 3-room | 234 | 518.91 | +51.58 | 3.2% |
| q3_2024 | ang mo kio / 3-room | 313 | 435.27 | +56.25 | 3.0% |
| q3_2024 | bukit merah / 3-room | 223 | 511.73 | +65.35 | 2.9% |
| q3_2024 | tampines / 4-room | 221 | 489.87 | +23.09 | 2.7% |
| q4_2024 | jurong west / 5-room | 181 | 550.96 | -16.93 | 3.4% |
| q4_2024 | woodlands / 4-room | 189 | 510.46 | +64.61 | 3.0% |
| q4_2024 | ang mo kio / 3-room | 316 | 383.19 | -39.79 | 2.9% |
| q4_2024 | bukit merah / 4-room | 172 | 506.72 | -15.55 | 2.7% |
| q4_2024 | yishun / 4-room | 171 | 500.09 | -9.07 | 2.6% |
| q4_2024 | jurong west / 4-room | 210 | 448.12 | -38.57 | 2.6% |

## Where the errors concentrate

- Pooling the two 2024 backtests, 4-room flats contribute 37.45% of total squared error, 3-room 29.22% and 5-room 23.18%, together 89.85%. This reflects both sample size and per-row error, so it does not mean these flat types are inherently hardest to predict.
- In the town by flat-type grouping, squared group mean bias is only 4.25%/2.80% of Q3/Q4 total SSE. Even removing every group's mean bias after the fact leaves most within-group variation; this is not an irreducible-noise bound and does not rule out finer features explaining it.
- The worst 5% of rows contribute 41.47% of SSE in Q3, 45.30% in Q4 and 43.15% in 2025 Q1. This grouping uses known validation errors, so it is diagnostic only and cannot route or filter test rows by unknown true rent.
- Jurong West 5-room flats are underpredicted by 223.52 in Q3 but only 16.93 in Q4. Bias changes over time, so adding or subtracting a fixed rent based on one quarter is unreliable.
- To keep lowering the score, next check the observable features, location detail and training coverage of large-error rows, and try transferable neighbour rent distributions, spatial clustering or target transforms rather than removing hard validation rows. New attempts must be recorded separately and compared on earlier windows first.

## Independent review

59 tests pass. `inspect_segment_experiments.py` independently checks 58 scores and row counts and SSE conservation for 96 groupings,
reloads the 14 new native models to check row-level predictions, and confirms that the original-model control in all four windows matches the previous export.
The record is `output_validation.json`. The main validation was not used for selection in this round; all reused windows still carry the repeated-viewing caveat.

## Method and leakage rules

- RMSE, MAE, bias, P90 absolute error, SSE share and large-error counts are computed by town, flat type, region, floor area, flat age, month, historical address sample count and their crosses.
- Correctors use only expanding-time OOF residuals from the six months before the prediction window. Each calibration prediction's base model uses only labels before that quarter; in-sample training errors are never used.
- Constant corrections are shrunk towards zero; trend corrections fit monthly within-group OOF residuals with slopes capped at +/-25 per month, new-window growth decays with a six-month time constant, and the final correction is capped at +/-200.
- Small groups are not corrected and unseen groups fall back to the global model. Segment models are chosen for at most four flat types by historical OOF SSE; each needs at least 3000 outer training rows and 100 inner validation rows, and rounds are chosen on an inner time split only.
- Segment and macro LightGBM models share one regularised 15-leaf configuration and are compared with the current option at 25%/50%/100% weights.
- Compares same-month COE, COE plus stocks, and COE plus stocks lagged two months and frozen at the cut-off. Backtests with same-period auxiliary data are not strict real-time forecasts; validation rent labels are never used.
- Each stage greedily selects on pooled 2024 Q3/Q4 SSE and always keeps the previous option. 2025 Q1 and six-month windows are computed only after locking; all periods have been viewed before and are not fresh independent tests.
- The twelve-month window is not rerun here; old twelve-month results cannot be transferred to new options, and local gains do not guarantee a Kaggle gain.
- Raw data, earlier experiment directories and submitted files are not overwritten; row-level labels, predictions and models stay local.
