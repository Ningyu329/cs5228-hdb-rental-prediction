# Keras Categorical-Embedding Neural Network Experiment

## Conclusion

On 2026-10-23, temporal backtests of two small networks with three random seeds each were completed. Neither network alone beats the existing approach.
The option selected on the earlier quarters, "75% original trend blend + 25% trend-residual network", lowers the main validation RMSE from 448.3346 to 447.7987,
but raises the full-year 2024 backtest from 491.7916 to 500.5986, so it is not a stable improvement. The original model and the public best score are therefore kept;
the competition test set was not read, and no submission was generated, submitted to Kaggle or pushed to GitHub.

## Setup

- Data: the 150,000-row cleaned training table with location features, with the same row order and labels as earlier experiments; hard rows are not removed.
- Environment: Python 3.12.14, Keras 3.15.1, JAX/JAXlib 0.12.0.1, Apple arm64 CPU.
- Inputs: the existing `housing` feature set, including town, street, block address, flat type, HDB attributes, and time and housing-derived features; no new macro or public data are added.
- Preprocessing: category vocabularies, numeric medians and scaling parameters are built from the current training part only, and numeric missing indicators are added. Categories seen fewer than twice and unknown categories map to index 0. Labels, ids and original row numbers are never features.
- Categorical embedding size is `min(16, max(2, ceil(sqrt(number of categories))))`; embeddings are concatenated with numeric features and fed into two ReLU layers of 128 and 64 units, with dropout 0.1.
- Dense-layer L2 is 0.00001 and embedding L2 0.000001; Adam with learning rate 0.002, batch size 1024, at most 80 epochs.
- The target is centred and scaled on the training part, with MSE loss. Predictions are converted back to currency before computing metrics; no log-rent loss is used.
- The direct network predicts rent; the trend-residual network first fits the `damped24` monthly market level on the training period, then predicts the static housing residual, with date and deterministic age inputs removed.
- Random seeds are fixed at 42, 123 and 2026, with no picking of the best seed; the three predictions are averaged with equal weight. Negative rent predictions are clipped to zero.
- Inside each outer training set, the last three months are used for early stopping on the standardised MSE without the L2 term, with patience 10. The network is then retrained from scratch on all outer training data with the inner best epoch count. The outer validation is never used to choose epochs, vocabularies or scaling parameters.

## Validation and results

All RMSE values are in SGD per month; lower is better. The direct and residual networks are means of three seeds' predictions, not means of three RMSEs.

| Validation task | Original trend blend | Direct network | Trend-residual network | Locked 25% residual-network blend |
| --- | ---: | ---: | ---: | ---: |
| Two earlier quarters pooled | 459.1748 | 463.0439 | 470.9556 | 458.0168 |
| 2025 Q1 | 448.3346 | 455.3848 | 453.9356 | 447.7987 |
| Second half of 2024, single 6-month forecast | 460.0744 | 463.6368 | 477.8421 | 459.1214 |
| Full year 2024, single 12-month forecast | 491.7916 | 593.4904 | 580.5380 | 500.5986 |

"Earlier quarters" means training only on data before 2024 Q3 and before 2024 Q4 respectively, validating on each quarter, and pooling squared errors.
Candidate weights are restricted in advance to 10%, 25%, 50% and 100%. Replacement requires a pooled RMSE gain of at least 1 point with neither quarter worsening by more than 1 point; nothing is deployed automatically.
`trend_blend25` improves the earlier quarters by 1.1580, by 0.2598 and 2.1858 in the two quarters respectively, passing this round's early eligibility threshold.
The candidate and weight were locked before the main validation and long-horizon metrics were viewed, and were not changed afterwards.

Main validation RMSE improves by 0.5359 (about 0.12%), and MAE falls from 326.4557 to 323.8454;
the 6-month RMSE improves by 0.9531; the 12-month RMSE worsens by 8.8070, with mean overprediction rising from 39.6462 to 104.9742.
These windows have been viewed repeatedly in earlier experiments and partly overlap, so they are not five independent or untouched test tasks.
Q3 and the 6-month task share a training origin but differ in query horizon and were retrained separately; they cannot be treated as two independent replications.

## Stability and error analysis

- Main validation RMSE of the direct network's three seeds is 462.4919, 463.4965 and 455.1618, and of the trend-residual network 456.6204, 454.0740 and 455.5298. Reporting only the best seed would be misleading.
- On the full-year backtest the direct network's three seeds give 562.4965, 603.2390 and 628.1100, so long-horizon performance is sensitive to initialisation; the trend-residual network is also above 578 for every seed, so the problem is not a single unlucky initialisation.
- The direct and trend-residual networks' main-validation residuals correlate with the original model at 0.9846 and 0.9827. Errors are highly correlated, so low-weight blends can only cancel a little.
- Inner best epochs are 1–13, never reaching the 80-epoch cap. Lower training loss does not consistently turn into lower future-window error, so more epochs would not solve the problem.
- The current monthly-level plus static-residual design does not solve cross-year market extrapolation, and overprediction grows clearly on the full-year backtest. This only shows these two recipes do not win consistently; it does not prove all neural architectures are ineffective or that RMSE has a mathematical floor of 448.

## Delivery and verification

30 outer training records, 30 option metrics and 30 single-seed metrics are saved, including repeated retraining from the same origin.
Code and dependency fingerprints, early-stopping histories, the selection lock, and model and prediction checksums are recorded. All 94 project tests pass.
An independent check reloads all 30 models, aligns original row numbers and labels, recomputes all metrics and three-seed means, verifies that preprocessing uses training data only,
and confirms that changing query labels or ids does not change predictions; the checksums of the two existing submission files are unchanged.

- [Option results](results.csv), [single-seed results](seed_results.csv), [diagnostic blends of the locked option by seed](seed_blends.csv).
- [Selection comparison](selection.csv), [selection lock](selection_locked.json), [monthly errors](monthly_errors.csv).
- [Independent check](output_validation.json), [training details](fit_details.json), [configuration and source fingerprints](experiment_manifest.json).
- Private row-level predictions and `.keras` models are stored locally and not committed to Git.

## Reproduction

First rebuild the location features and the original trend blend's row-level reference predictions as described in the README. With different dependencies or source bytes, use a new output directory instead of rewriting historical fingerprints to reuse caches.

```bash
uv pip install --python .venv/bin/python -r requirements-keras.txt
.venv/bin/python src/models/run_keras_experiments.py --output-dir reports/experiments/keras_reproduction
.venv/bin/python src/models/inspect_keras_experiments.py --output-dir reports/experiments/keras_reproduction
.venv/bin/python -m unittest discover -s tests -v
```

The directory name can change, but the inputs and source dependencies must stay the same. Model caches are isolated by experiment fingerprint and only resume when fingerprints match.
The implementation follows the official Keras documentation on [backend configuration](https://keras.io/getting_started/), [early stopping](https://keras.io/api/callbacks/early_stopping/) and [random seeds](https://keras.io/api/utils/python_utils/).
