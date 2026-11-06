# Identifiability of Large Overpredictions

A fixed LightGBM binary diagnostic predicts which rows the original model will overpredict by more than 1000; rent predictions are not modified. Training targets come from strictly earlier OOF errors.

| Window | Inputs | Share overpredicted >1000 | ROC-AUC | AP | Precision in top-5% risk | Recall of positives | Lift over random |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| q3_2024 | housing | 3.59% | 0.501 | 0.037 | 4.32% | 6.03% | 1.20x |
| q3_2024 | housing_history_control | 3.59% | 0.510 | 0.039 | 4.55% | 6.35% | 1.27x |
| q4_2024 | housing | 3.34% | 0.491 | 0.033 | 3.38% | 5.07% | 1.01x |
| q4_2024 | housing_history_control | 3.34% | 0.499 | 0.033 | 3.38% | 5.07% | 1.01x |
| primary_2025 | housing | 3.45% | 0.521 | 0.036 | 3.36% | 4.87% | 0.97x |
| primary_2025 | housing_history_control | 3.45% | 0.528 | 0.036 | 2.24% | 3.25% | 0.65x |

AP is average precision. The expected AP of a random ranking is close to the positive share; the top-5% risk set is a fixed alert budget, so no true validation label is used to pick a threshold. Metric definitions follow the [scikit-learn evaluation documentation](https://scikit-learn.org/stable/modules/model_evaluation.html#precision-recall-f-measure-metrics)。

History-extended inputs include the original prediction, the mean/median/std/quantiles of same-block same-flat-type rents over the past twelve months, and history coverage. Each historical OOF quarter is built only from earlier training data.

This experiment only tests this fixed diagnostic; it does not prove that no feature or model can identify cheap rentals, and the probabilities are not used to correct rent predictions. True rent bins, errors and row ids are never inputs.

Conclusion: across the six diagnostics ROC-AUC is 0.491–0.529 and AP 0.033–0.039, with no stable identification ability across windows. In particular, with history-extended inputs the top-5% risk rows on the main validation have only 2.24% precision and 3.25% recall of positives, which does not support an automatic two-stage price reduction. The failure of this fixed model does not prove that the information is absent or that a score of 200 is unreachable.

The same parameters, features and windows reproduce the six scores after fresh training; the probabilities were not written into or used to overwrite the original rent model or submission files.
