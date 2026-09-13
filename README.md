# CS5228 HDB Rental Prediction

Private team repository for the CS5228-2610 Kaggle competition.

## Objective

Predict `MONTHLY_RENT` for Singapore HDB flats and evaluate models using RMSE.

The main approach is supervised regression:

`flat attributes + location + time + auxiliary information → monthly rent`

Clustering and other data-mining methods may support exploration or feature engineering, but they are not the primary prediction method.

## Project stages

### 1. Data audit and cleaning

- Inspect `train.csv` and `test.csv` for types, missing values, duplicates, anomalies, and train/test differences.
- Standardise text fields such as town, block, street, flat type, flat model, and furnishing status.
- Parse rental approval dates, normalise numerical columns, and derive basic fields such as flat age and fee indicators.
- Clean all auxiliary sources before use:
  - HDB block addresses, coordinates, and invalid completion years;
  - MRT station coordinates and station status;
  - school and mall coordinates;
  - COE currency values and monthly rounds;
  - stock-price dates and monthly aggregates.
- Preserve `data/raw/` unchanged and make each cleaning step reproducible by script.

### 2. Exploratory analysis and baseline

- Analyse rent distributions, time trends, town and flat-type differences, floor area, and flat age.
- Use a time-based validation split; retain a random split only as a secondary reference.
- Build simple baselines, followed by a supervised regression baseline such as CatBoost or LightGBM using main-table features.
- Record validation RMSE, model configuration, feature set, and create a correctly formatted test prediction file.

### 3. Midterm progress-report slides

- Present the problem definition, dataset summary, and data-cleaning decisions.
- Show key EDA findings, validation design, and baseline RMSE.
- Describe the auxiliary datasets and the planned location-feature approach.
- State current limitations, risks of data leakage, and next steps.

### 4. Address matching and location features

- Match `TOWN + BLOCK + STREET` to the HDB block data and report matching coverage.
- Add coordinates, planning area, region, completion year, and match-status indicators.
- Create distance and density features for open MRT stations, schools, and shopping malls.
- Optionally test geographic clustering as an additional regression feature.

### 5. Feature expansion and model improvement

- Add time, flat-age, fee, address-level, and location-derived features.
- Evaluate monthly COE features and carefully selected aggregated market indicators.
- Compare CatBoost, LightGBM/XGBoost, and an optional ensemble on the same validation split.
- Use feature importance and segment-level errors to guide further improvements.

### 6. Final evaluation and delivery

- Perform ablation experiments for main-table, location, facilities, and macro features.
- Analyse errors by town, flat type, age, size, time period, and address-match status.
- Train the chosen final approach, generate the Kaggle submission, and document results.
- Complete the final report with methodology, model comparison, findings, limitations, and reproducibility notes.

## Repository layout

- `src/data/`: cleaning and data-integration scripts
- `src/features/`: feature-engineering scripts
- `src/models/`: training, evaluation, and prediction scripts
- `notebooks/`: exploratory analysis only
- `configs/`: shared model and pipeline settings
- `reports/`: figures, progress-report slides, and written reports

Raw Kaggle competition data, local models, and submission files are excluded from Git. Each authorised team member should obtain competition data through Kaggle.

## Collaboration workflow

1. Create a focused branch from `main`.
2. Keep changes reproducible and run relevant checks locally.
3. Open a pull request for review before merging into `main`.
4. Record every experiment's data version, validation strategy, feature set, model configuration, and RMSE.
