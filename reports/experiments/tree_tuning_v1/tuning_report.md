# CatBoost / LightGBM temporal tuning

## Design

- Fixed housing feature set: no new target statistics, MRT, macro data, test data or label removal.
- Screen all eight configurations on Apr-Jun 2024, training through Mar 2024. Carry one new configuration per family forward.
- Compare challengers, original components and 25/50/75% CatBoost blends on Jul-Sep and Oct-Dec 2024. Choose by pooled squared error.
- The original 50/50 blend remains eligible. Both earlier windows are tuning data, not unbiased final performance estimates.
- Each refit uses an internal three-month early-stopping window, never outer-query labels. The first screening fit is reused only for the identical inner window.
- Lock before Jan-Mar 2025 evaluation. This primary period has been inspected in earlier project work and is not a pristine final test.
- Search is bounded, not exhaustive; increased caps/depths do not guarantee improvement. No competition submission is generated.

## Screening

| Candidate | RMSE | Rounds | Cap |
| --- | ---: | ---: | ---: |
| catboost_d8_regularized | 503.1955 | 770 | 800 |
| catboost_d6_long | 503.6221 | 1200 | 1200 |
| catboost_original | 504.7742 | 400 | 400 |
| lightgbm_15_regularized | 507.5195 | 367 | 1000 |
| lightgbm_original | 510.0317 | 187 | 400 |
| lightgbm_31_slow | 510.0581 | 342 | 1200 |
| lightgbm_63_flexible | 513.5735 | 118 | 1000 |
| lightgbm_63_regularized | 513.6742 | 118 | 1000 |

## Locked selection

Selected: `blend__catboost_d8_regularized__lightgbm_15_regularized__cat50`.

```json
{
  "catboost_d8_regularized": 0.5,
  "lightgbm_15_regularized": 0.5
}
```

## Before / after

| Model | Earlier pooled RMSE | Primary RMSE | Primary MAE | Primary bias |
| --- | ---: | ---: | ---: | ---: |
| Original CatBoost | 464.9071 | 457.2788 | 336.0324 | -34.63 |
| Tuned CatBoost | 462.3785 | 455.2215 | 337.1002 | -50.00 |
| Original LightGBM | 467.9085 | 455.1987 | 332.9281 | -26.55 |
| Tuned LightGBM | 465.5564 | 453.0133 | 331.1734 | -24.77 |
| Original 50/50 blend | 461.8684 | 452.1702 | 331.1601 | -30.59 |
| Locked selection | 460.4831 | 450.4735 | 331.1260 | -37.38 |

Primary RMSE change (old minus selected): +1.6967 (+0.38%).
A negative change means the locked tuning choice did not improve the primary window; do not reselect based on that observation.

## Limits and reproducibility

- Static HDB snapshot availability limitations from stage 5 remain. Missing apartment-level details and long-horizon market uncertainty are unchanged.
- A model selected at its iteration cap may still benefit from a later higher-cap experiment; this run makes no global-optimum claim.
- Random splits are deliberately omitted: this experiment targets future-month prediction.
- Saved primary models are diagnostic fits through Dec 2024, not final full-data models.
- See experiment_manifest.json for parameters, input/code hashes, library versions, fixed search/selection rules, and reference artifact hashes.
- See screening.csv, backtest_results.csv, selection_backtests.csv, selection_locked.json, results.csv, comparison.csv and group_errors.csv.

[CatBoost parameter tuning](https://catboost.ai/docs/en/concepts/parameter-tuning) · [LightGBM parameter tuning](https://lightgbm.readthedocs.io/en/stable/Parameters-Tuning.html)
