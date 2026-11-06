# Data Quality After Cleaning

This report compares the raw data in data/raw/ with the cleaned output in data/interim/. Cleaning removed no records and did not modify the raw files.

| Dataset | Raw rows | Clean rows | Raw missing cells | Clean missing cells | Raw duplicate rows | Clean duplicate rows |
| --- | --- | --- | --- | --- | --- | --- |
| Training table | 150,000 | 150,000 | 0 | 1 | 425 | 641 |
| Test table | 50,000 | 50,000 | 0 | 0 | 5,508 | 7,764 |
| HDB blocks | 9,464 | 9,464 | 0 | 11 | 0 | 0 |
| MRT stations | 243 | 243 | 0 | 0 | 0 | 0 |
| Schools | 337 | 337 | 0 | 0 | 0 | 0 |
| Shopping malls | 113 | 113 | 0 | 0 | 0 | 0 |
| COE prices | 539 | 539 | 0 | 0 | 0 | 0 |
| Stock prices | 91,642 | 91,642 | 510 | 510 | 0 | 0 |

## 1. Processing completed

- Added RENT_YEAR, RENT_MONTH, RENT_MONTH_INDEX and FLAT_AGE to the main tables for baseline models and temporal validation.
- Standardised FLAT_TYPE: training categories reduced from 11 to 6; test categories from 6 to 6. Variants such as 3 room and 3-room were unified.
- Town, block, street, flat model and furnishing fields were lowercased, trimmed and had repeated spaces collapsed.
- Complete duplicate rows increased after cleaning because records that differed only in formatting became identical; no record was removed or merged.
- 1 training records have a rental year before the lease commencement year; their FLAT_AGE is left missing to avoid negative ages.
- 11 HDB YEAR_COMPLETED values of -1 were converted to missing; the cleaned field has 11 missing values.
- 539 COE premium values with currency symbols or commas were converted to numbers; 539/539 rows received MONTH_NUM and YEAR_MONTH.
- MRT, school, mall and stock data had text, date and numeric formats standardised, without aggregation or deletion.

## 2. Consistency and usability checks

- Cleaned training address matches: 150,000/150,000 (100.0%).
- Cleaned test address matches: 50,000/50,000 (100.0%).
- Cleaned MRT status: 231 open, 12 planned. The first location features should use open stations only.
- The stock table still has 85 rows with missing closing prices; these are missing in the raw data and are handled when aggregating by security and month.

## 3. Remaining limitations and next steps

- FEE is always 0 and FURNISHED is always yes; both are kept for lineage but should be excluded from the first model.
- Duplicates are not removed. Identical inputs can have different rents, so deduplicating on features would discard valid observations.
- COE can have several rounds for the same month and category; feature engineering must choose the last round, the mean or another aggregation.
- Stock data are only cleaned here; monthly returns and volatility are built in stage 5 if validation supports them.
- Planned MRT stations may leak future information; they should not be used for early rental records without reliable opening dates.
- Next step: build a time-split baseline on data/interim/train_clean.csv and data/interim/test_clean.csv using main-table features only.

## 4. Input and output paths

- Input: data/raw/, read-only Kaggle files.
- This stage: data/interim/, cleaned main and auxiliary tables.
- Later feature engineering: data/processed/, merged modelling tables.
- This report: reports/data_quality_after_cleaning.md.

## 5. Train/test distribution comparison

These tables identify structural differences between training and test data. This is an audit only and does not analyse rent; rent distributions and trends are covered by the stage 2 EDA.

### Rental year

| RENT_YEAR | Train count | Train share | Test count | Test share | Share difference (test - train) |
| --- | --- | --- | --- | --- | --- |
| 2021 | 37,619 | 25.1% | 0 | 0.0% | -25.1% |
| 2022 | 32,564 | 21.7% | 0 | 0.0% | -21.7% |
| 2023 | 35,654 | 23.8% | 0 | 0.0% | -23.8% |
| 2024 | 35,239 | 23.5% | 0 | 0.0% | -23.5% |
| 2025 | 8,924 | 5.9% | 28,765 | 57.5% | +51.6% |
| 2026 | 0 | 0.0% | 21,235 | 42.5% | +42.5% |

### Flat type

| FLAT_TYPE | Train count | Train share | Test count | Test share | Share difference (test - train) |
| --- | --- | --- | --- | --- | --- |
| 1-room | 42 | 0.0% | 12 | 0.0% | -0.0% |
| 2-room | 3,098 | 2.1% | 1,436 | 2.9% | +0.8% |
| 3-room | 49,024 | 32.7% | 16,425 | 32.9% | +0.2% |
| 4-room | 54,079 | 36.1% | 18,045 | 36.1% | +0.0% |
| 5-room | 35,523 | 23.7% | 11,530 | 23.1% | -0.6% |
| executive | 8,234 | 5.5% | 2,552 | 5.1% | -0.4% |

### Town

| TOWN | Train count | Train share | Test count | Test share | Share difference (test - train) |
| --- | --- | --- | --- | --- | --- |
| ang mo kio | 8,283 | 5.5% | 2,664 | 5.3% | -0.2% |
| bedok | 8,907 | 5.9% | 2,803 | 5.6% | -0.3% |
| bishan | 3,306 | 2.2% | 982 | 2.0% | -0.2% |
| bukit batok | 5,605 | 3.7% | 2,057 | 4.1% | +0.4% |
| bukit merah | 8,270 | 5.5% | 2,832 | 5.7% | +0.2% |
| bukit panjang | 3,717 | 2.5% | 1,228 | 2.5% | -0.0% |
| bukit timah | 438 | 0.3% | 156 | 0.3% | +0.0% |
| central | 2,278 | 1.5% | 831 | 1.7% | +0.1% |
| choa chu kang | 5,300 | 3.5% | 1,687 | 3.4% | -0.2% |
| clementi | 5,429 | 3.6% | 1,883 | 3.8% | +0.1% |
| geylang | 4,504 | 3.0% | 1,445 | 2.9% | -0.1% |
| hougang | 7,115 | 4.7% | 2,298 | 4.6% | -0.1% |
| jurong east | 4,024 | 2.7% | 1,371 | 2.7% | +0.1% |
| jurong west | 10,482 | 7.0% | 3,663 | 7.3% | +0.3% |
| kallang/whampoa | 5,689 | 3.8% | 1,916 | 3.8% | +0.0% |
| marine parade | 1,637 | 1.1% | 493 | 1.0% | -0.1% |
| pasir ris | 4,089 | 2.7% | 1,329 | 2.7% | -0.1% |
| punggol | 6,668 | 4.4% | 2,286 | 4.6% | +0.1% |
| queenstown | 6,115 | 4.1% | 2,148 | 4.3% | +0.2% |
| sembawang | 3,553 | 2.4% | 1,298 | 2.6% | +0.2% |
| sengkang | 9,773 | 6.5% | 3,231 | 6.5% | -0.1% |
| serangoon | 3,398 | 2.3% | 1,099 | 2.2% | -0.1% |
| tampines | 9,942 | 6.6% | 3,198 | 6.4% | -0.2% |
| toa payoh | 5,988 | 4.0% | 2,055 | 4.1% | +0.1% |
| woodlands | 7,577 | 5.1% | 2,385 | 4.8% | -0.3% |
| yishun | 7,913 | 5.3% | 2,662 | 5.3% | +0.0% |

### Floor area and flat age

| Numerical field | Train min | Train median | Train max | Test min | Test median | Test max |
| --- | --- | --- | --- | --- | --- | --- |
| FLOOR_AREA_SQM | 31.0 | 92.0 | 215.0 | 31.0 | 92.0 | 215.0 |
| FLAT_AGE | 0.0 | 34.0 | 59.0 | 3.0 | 37.0 | 59.0 |

The test set covers later rental years, so baselines must use a temporal split rather than only a random split. Town, flat type, floor area and age are examined against rent in the EDA.
