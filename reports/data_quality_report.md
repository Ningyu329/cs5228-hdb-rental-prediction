# Data Quality Audit Report

Audit date: 2026-09-24  
Scope: `data/raw/train.csv`, `data/raw/test.csv` and every CSV under `data/raw/auxiliary/`.  
This audit only reads and diagnoses the raw data; no CSV file was modified.

## 1. Summary

The main data are complete: no field has missing values, and all date and numeric fields parse. The
issues to handle first are:

1. `FLAT_TYPE` in the training set mixes format variants such as `4 room` and `4-room`, which need
   standardising;
2. `FEE` is always 0 and `FURNISHED` is always `yes`, so neither column currently discriminates;
3. many training records share identical inputs but have different `MONTHLY_RENT`, so rows cannot be
   deduplicated on features;
4. 85 stock rows are missing all price values;
5. COE data usually have two bidding rounds per month and category, so a monthly aggregation rule is
   needed before merging.

Address quality is good: matching the main data to the HDB block table on `TOWN + BLOCK + STREET`
gives a 100% match rate, a reliable basis for later geographic features.

## 2. Main data overview

| Dataset | Rows | Fields | Date range | Missing values | Complete duplicate rows |
|---|---:|---:|---|---:|---:|
| `train.csv` | 150,000 | 11 | 2021-01 to 2025-03 | 0 | 425 |
| `test.csv` | 50,000 | 10 | 2025-03 to 2026-07 | 0 | 5,508 |

The only field in `train.csv` that is not in `test.csv` is the target `MONTHLY_RENT`; the input fields
are identical.

### 2.1 Field validity

| Field | Result | Notes |
|---|---|---|
| `RENT_APPROVAL_DATE` | All parse | Format `YYYY-MM`. Both train and test contain 2025-03, so temporal validation must handle this overlapping month carefully. |
| `LEASE_COMMENCE_DATE` | All parse | Range 1966–2022; usable for flat age. |
| `FLOOR_AREA_SQM` | All parse | Range 31–215 sqm; no implausible negative or zero values. |
| `FEE` | All 0 | Constant feature; can be removed from the first model, and the report should state it has no variation. |
| `FURNISHED` | All `yes` | Constant categorical feature; can be removed from the first model. |
| `MONTHLY_RENT` | All parse | Training set only; range 300–7,600. |

No text field has leading or trailing spaces.

### 2.2 Category formats and train/test consistency

- `TOWN`: 26 towns in both train and test, with identical category sets.
- `STREET`: 576 in train, 573 in test. Test has no street unseen in train; train has 3 streets absent
  from test, a normal temporal-split difference.
- `FLAT_TYPE`: 11 raw values in train, 6 in test. Train additionally contains `1 room`, `2 room`,
  `3 room`, `4 room` and `5 room`, while test uses the hyphenated form. They should be unified as
  `1-room`, `2-room`, `3-room`, `4-room` and `5-room`.
- `FLAT_MODEL`: train only adds one `improved maisonette` category, which is rare but valid and should
  not be removed as a format error.
- `BLOCK`: each set has a few block numbers absent from the other, a normal difference between old and
  new samples.

### 2.3 Interpreting duplicates

The training set has 425 complete duplicate rows and the test set 5,508. More importantly, the
training set has 13,434 groups of identical input features with 15,547 extra rows, and 13,191 of these
groups have several different rent labels.

Rows therefore cannot simply be deduplicated on input fields. The same town, block, street, flat type,
area and date can still have different rents, which shows that factors outside the data also affect
rent. The 425 complete duplicates should only be flagged for review; no record should be deleted
before a validation scheme exists.

## 3. Auxiliary data overview

| File | Rows | Fields | Missing / duplicates | Cleaning focus |
|---|---:|---:|---|---|
| `sg-hdb-block.csv` | 9,464 | 11 | No missing values or complete duplicates | 11 rows with `YEAR_COMPLETED=-1` should become missing. |
| `sg-mrt-stations.csv` | 243 | 8 | No missing values or duplicate station codes | 231 open and 12 planned stations; the first version uses open stations only. |
| `sg-schools.csv` | 337 | 14 | No missing values or duplicate school names | All coordinates parse; usable for distance or density features. |
| `sg-shopping-malls.csv` | 113 | 3 | No missing values, duplicate names or coordinates | All coordinates parse; usable for distance or density features. |
| `sg-coe-prices.csv` | 539 | 8 | No missing values or complete duplicates | Currency and comma-formatted values need converting; the same month and category often has two rounds. |
| `sg-stock-prices.csv` | 91,642 | 9 | 85 rows missing all price values; no duplicate security-date pairs | Dates 2021-01-04 to 2026-07-30; needs monthly aggregation. |

### 3.1 HDB block data

- Matching on lowercased, trimmed `TOWN + BLOCK + STREET` gives 150,000/150,000 (100%) for train and
  50,000/50,000 (100%) for test.
- Address keys are unique, and all coordinates parse and lie within valid latitude/longitude ranges.
- The 11 rows with `YEAR_COMPLETED=-1` are not valid completion years and should become missing in the
  cleaning layer, to avoid implausible building ages.

### 3.2 MRT, school and mall data

- MRT `STATUS` is `open` (231) or `planned` (12).
- Planned MRT stations lack historical opening dates, so using them for early rental records could leak
  future information; the first distance features use `open` stations only.
- The school and mall tables have no missing or duplicate coordinates and can be used for distance
  calculations after cleaning.

### 3.3 COE data

- Covers 2021-01 to 2026-08; categories a, b, c and e.
- `QUOTA_PREMIUM` and `PREVAILING_QUOTA_PREMIUM` contain currency symbols and commas; `BIDS_RECEIVED`
  also contains commas. All need converting to numbers.
- All amounts, quotas and bid counts parse after format cleaning.
- 268 month-category combinations have several rounds. Before joining to the main table, a rule is
  needed: last round, mean, or separate encoding.
- 2026 only covers January–August, compatible with the test horizon to 2026-07, but features may only
  use information available in the target month.

### 3.4 Stock data

- 77 companies and 77 security codes, from 2021-01-04 to 2026-07-30.
- `OPEN`, `HIGH`, `LOW`, `CLOSE`, `ADJUSTED_CLOSE` and `VOLUME` each have 85 missing values,
  concentrated on 2026-05-01 and 2026-06-01.
- All existing values parse and there are no duplicate security-date pairs.
- Daily per-stock fields should not be merged directly into the main table; a few monthly market
  indicators, such as monthly returns or volatility, should be built first.

## 4. Recommended cleaning priorities

1. Keep `data/raw/` unchanged and write cleaned outputs to `data/interim/`.
2. Standardise `FLAT_TYPE` in the main tables and unify all text fields used for address matching.
3. Normalise dates, numbers and basic features such as flat age; drop the constant columns `FEE` and
   `FURNISHED` from the first model.
4. Convert invalid completion years in the HDB block table to missing and build a reusable
   address-matching script.
5. Clean COE values and months; merge only after defining a monthly aggregation rule.
6. Define a per-security, per-date rule for missing stock values and aggregate monthly later.
7. Build a baseline on the cleaned main table first, then compare the RMSE gains from geographic,
   amenity and macro features step by step.

## 5. Conclusion

The data need no large-scale imputation. The first cleaning version should focus on unifying category
formats, handling invalid codes, converting dates and numbers, and avoiding temporal leakage. The HDB
address table is of high quality and matches the main tables completely, so it is the first data
source for extending geographic features after the mid-term.
