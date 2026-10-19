"""Preserve formatting as an extra view, only after exact row alignment checks."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.clean_data import clean_main_rows

RAW_CATS = ["RAW_STREET", "RAW_FLAT_TYPE"]
RAW_NUMS = ["RAW_STREET_CASE_CHANGED", "RAW_FLAT_FORMAT_CHANGED"]


def verify_alignment(raw: pd.DataFrame, clean: pd.DataFrame) -> list[dict]:
    if len(raw) != len(clean) or not raw.index.equals(clean.index):
        raise ValueError("Raw and cleaned row counts/order indices differ")
    canonical, _ = clean_main_rows(raw.fillna("").astype(str).to_dict("records"),
                                   has_target="MONTHLY_RENT" in raw)
    expected = pd.DataFrame(canonical, index=raw.index)
    rows = []
    numeric = {"FLOOR_AREA_SQM", "LEASE_COMMENCE_DATE", "FEE", "MONTHLY_RENT",
               "RENT_YEAR", "RENT_MONTH", "RENT_MONTH_INDEX", "FLAT_AGE"}
    for col in expected:
        if col not in clean:
            raise ValueError(f"Missing cleaned field: {col}")
        if col in numeric:
            left = pd.to_numeric(expected[col].replace("", np.nan))
            right = pd.to_numeric(clean[col].replace("", np.nan))
            equal = left.eq(right) | (left.isna() & right.isna())
        else:
            equal = expected[col].astype(str).eq(clean[col].fillna("").astype(str))
        if not equal.all():
            raise ValueError(f"Raw/clean values or row order disagree: {col}")
        changed = int(raw[col].astype(str).ne(clean[col].astype(str)).sum()) if col in raw else 0
        rows.append(dict(field=col, aligned=True, raw_distinct=int(raw[col].nunique()) if col in raw else None,
                         clean_distinct=int(clean[col].nunique()), textual_changes=changed))
    return rows


def attach_raw_view(frame: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    """No positional attach without validating the underlying core feature values."""
    keys = ["RENT_APPROVAL_DATE", "TOWN", "BLOCK", "STREET", "FLAT_TYPE", "FLAT_MODEL",
            "FLOOR_AREA_SQM", "LEASE_COMMENCE_DATE", "FEE", "FURNISHED"]
    core = frame[keys].copy()
    core["RENT_APPROVAL_DATE"] = pd.to_datetime(core.RENT_APPROVAL_DATE).dt.strftime("%Y-%m")
    canonical, _ = clean_main_rows(raw.fillna("").astype(str).to_dict("records"), has_target=False)
    expected = pd.DataFrame(canonical, index=raw.index)
    if len(core) != len(raw) or not core.index.equals(raw.index):
        raise ValueError("Raw rows do not match feature rows")
    for c in keys:
        if c in {"FLOOR_AREA_SQM", "LEASE_COMMENCE_DATE", "FEE"}:
            np.testing.assert_allclose(pd.to_numeric(core[c]), pd.to_numeric(expected[c]), rtol=0, atol=0)
        elif not core[c].astype(str).equals(expected[c].astype(str)):
            raise ValueError(f"Processed/raw row alignment failure: {c}")
    result = frame.copy()
    for c in ["STREET", "FLAT_TYPE"]:
        result[f"RAW_{c}"] = raw[c].astype(str)
    result[RAW_NUMS[0]] = raw.STREET.astype(str).ne(expected.STREET).astype(int)
    result[RAW_NUMS[1]] = raw.FLAT_TYPE.astype(str).ne(expected.FLAT_TYPE).astype(int)
    return result
