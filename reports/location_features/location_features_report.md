# Address Matching and Location Features Run Report

This report is generated from the inputs of this run only; historical audit match rates are not substituted.

| Dataset | Rows | Matched rows | Address match rate | Valid coordinate rows | Missing building age rows |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 150,000 | 150,000 | 100.00% | 150,000 | 159 |
| test | 50,000 | 50,000 | 100.00% | 50,000 | 0 |

## Features and processing rules

- Addresses are matched exactly on standardised TOWN + BLOCK + STREET; missing addresses are not matched and no fuzzy guessing is used.
- HDB address keys must be unique. Main-table row count, order, duplicates and labels are preserved; rent is never used to build location features.
- Coordinates, planning area, region, completion year, maximum floor and postal code are kept. ADDRESS_MATCHED and LOCATION_COORDINATES_VALID are flagged separately.
- BUILDING_AGE = rental approval year - YEAR_COMPLETED, which differs from FLAT_AGE based on lease commencement. Invalid, missing or negative building ages are left empty with a missing indicator.
- Distances use the Haversine great-circle distance in km. Facility counts within a radius (inclusive) are density proxies, not walking distance or service quality.
- MRT stations at the same coordinates count once, so interchanges with several line codes are not double counted; schools and malls are deduplicated by name and coordinates.
- Distances and counts are empty for invalid or unmatched coordinates. With no usable facility the nearest distance is empty and counts are 0 for valid coordinates.

## Temporal availability and limitations

- MRT snapshot mode only keeps stations with current STATUS=open, which does not prove they were open in historical rental months, so temporal leakage remains possible. Formal temporal validation should supply opening dates and use historical mode.
- School, mall and HDB area attributes come from static auxiliary tables without full historical versions. The location features cannot be claimed to remove temporal leakage entirely.
- Feature generation fits no imputer, encoder or clustering. Later modelling must fit such transforms on the training fold only.

## Auxiliary data checks

| Source | Input rows | Valid unique locations | Invalid coordinate rows |
| --- | ---: | ---: | ---: |
| mrt | 243 | 192 | 0 |
| school | 337 | 337 | 0 |
| mall | 113 | 113 | 0 |

## Address matching by town

| Dataset | Town | Rows | Matched rows |
| --- | --- | ---: | ---: |
| train | ang mo kio | 8283 | 8283 |
| train | bedok | 8907 | 8907 |
| train | bishan | 3306 | 3306 |
| train | bukit batok | 5605 | 5605 |
| train | bukit merah | 8270 | 8270 |
| train | bukit panjang | 3717 | 3717 |
| train | bukit timah | 438 | 438 |
| train | central | 2278 | 2278 |
| train | choa chu kang | 5300 | 5300 |
| train | clementi | 5429 | 5429 |
| train | geylang | 4504 | 4504 |
| train | hougang | 7115 | 7115 |
| train | jurong east | 4024 | 4024 |
| train | jurong west | 10482 | 10482 |
| train | kallang/whampoa | 5689 | 5689 |
| train | marine parade | 1637 | 1637 |
| train | pasir ris | 4089 | 4089 |
| train | punggol | 6668 | 6668 |
| train | queenstown | 6115 | 6115 |
| train | sembawang | 3553 | 3553 |
| train | sengkang | 9773 | 9773 |
| train | serangoon | 3398 | 3398 |
| train | tampines | 9942 | 9942 |
| train | toa payoh | 5988 | 5988 |
| train | woodlands | 7577 | 7577 |
| train | yishun | 7913 | 7913 |
| test | ang mo kio | 2664 | 2664 |
| test | bedok | 2803 | 2803 |
| test | bishan | 982 | 982 |
| test | bukit batok | 2057 | 2057 |
| test | bukit merah | 2832 | 2832 |
| test | bukit panjang | 1228 | 1228 |
| test | bukit timah | 156 | 156 |
| test | central | 831 | 831 |
| test | choa chu kang | 1687 | 1687 |
| test | clementi | 1883 | 1883 |
| test | geylang | 1445 | 1445 |
| test | hougang | 2298 | 2298 |
| test | jurong east | 1371 | 1371 |
| test | jurong west | 3663 | 3663 |
| test | kallang/whampoa | 1916 | 1916 |
| test | marine parade | 493 | 493 |
| test | pasir ris | 1329 | 1329 |
| test | punggol | 2286 | 2286 |
| test | queenstown | 2148 | 2148 |
| test | sembawang | 1298 | 1298 |
| test | sengkang | 3231 | 3231 |
| test | serangoon | 1099 | 1099 |
| test | tampines | 3198 | 3198 |
| test | toa payoh | 2055 | 2055 |
| test | woodlands | 2385 | 2385 |
| test | yishun | 2662 | 2662 |

## Next validation step

Compare validation RMSE of main-table features and these location features on the same temporal split. This script only builds features; it trains no model and makes no RMSE claim.
