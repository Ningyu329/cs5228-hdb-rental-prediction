# CS5228 HDB Rental Prediction

Private team repository for the CS5228-2610 Kaggle competition.

## Objective

Predict `MONTHLY_RENT` for Singapore HDB flats and evaluate models with RMSE.

## Repository layout

- `src/data/`: cleaning and data-integration scripts
- `src/features/`: feature-engineering scripts
- `src/models/`: training, evaluation, and prediction scripts
- `notebooks/`: exploratory analysis only
- `configs/`: shared model and pipeline settings
- `reports/`: figures and project reports

Raw Kaggle competition data, local models, and submission files are deliberately excluded from Git. Each authorised team member should obtain competition data through Kaggle.

## Team workflow

1. Create a focused branch from `main`.
2. Make changes and run the relevant checks locally.
3. Open a pull request for review before merging into `main`.
4. Record every model experiment, validation strategy, feature set, and RMSE.
