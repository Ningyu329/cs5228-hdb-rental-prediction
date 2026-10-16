"""Housing transformations and lagged, cutoff-frozen monthly macro features.

No rental labels enter these functions. Monthly market data is aggregated before
joining, using unadjusted CLOSE to avoid retrospective adjusted-price revisions.
Release timestamps are unavailable, so a two-month lag is a conservative proxy,
not proof of point-in-time data availability.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

COE_FEATURES = [f"COE_{category}_{metric}" for category in ["A", "B", "C", "E"] for metric in ["PREMIUM", "BID_QUOTA_RATIO"]]
STOCK_FEATURES = ["STOCK_EQ_WEIGHT_RETURN", "STOCK_MEAN_DAILY_VOL", "STOCK_POSITIVE_RETURN_SHARE", "STOCK_SYMBOL_COUNT"]
ENGINEERED_NUMERIC = ["LOG_FLOOR_AREA", "RENT_MONTH_SIN", "RENT_MONTH_COS", "FLAT_TYPE_ROOMS", "AREA_PER_ROOM"]
ENGINEERED_CATEGORICAL = ["BLOCK_ADDRESS", "FLAT_AGE_BAND"]


def add_housing_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    dates = pd.to_datetime(out["RENT_APPROVAL_DATE"])
    out["RENT_MONTH_SIN"] = np.sin(2 * np.pi * dates.dt.month / 12)
    out["RENT_MONTH_COS"] = np.cos(2 * np.pi * dates.dt.month / 12)
    area = pd.to_numeric(out["FLOOR_AREA_SQM"], errors="coerce")
    out["LOG_FLOOR_AREA"] = np.log1p(area.where(area.gt(0)))
    rooms = out["FLAT_TYPE"].astype("string").str.extract(r"^([1-5])-room$", expand=False)
    out["FLAT_TYPE_ROOMS"] = pd.to_numeric(rooms, errors="coerce").astype(float)
    out["AREA_PER_ROOM"] = area / out["FLAT_TYPE_ROOMS"]
    parts = [out[c].astype("string").fillna("missing").str.strip().str.lower() for c in ["TOWN", "BLOCK", "STREET"]]
    out["BLOCK_ADDRESS"] = parts[0] + "|" + parts[1] + "|" + parts[2]
    age = pd.to_numeric(out["FLAT_AGE"], errors="coerce")
    out["FLAT_AGE_BAND"] = pd.cut(age, [-np.inf, 10, 20, 30, 40, 50, np.inf], right=False).astype("string").fillna("missing")
    return out


def aggregate_macro(coe: pd.DataFrame, stock: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    coe = coe.copy()
    coe["_MONTH"] = pd.to_datetime(coe["YEAR_MONTH"], format="%Y-%m").dt.to_period("M")
    coe["CATEGORY"] = coe["CATEGORY"].astype(str).str.upper()
    for column in ["QUOTA_PREMIUM", "QUOTA", "BIDS_RECEIVED"]:
        coe[column] = pd.to_numeric(coe[column], errors="coerce")
    coe_parts = []
    for category in ["A", "B", "C", "E"]:
        monthly = coe.loc[coe.CATEGORY.eq(category)].groupby("_MONTH").agg(
            PREMIUM=("QUOTA_PREMIUM", "mean"), QUOTA=("QUOTA", "sum"), BIDS=("BIDS_RECEIVED", "sum"),
        )
        monthly[f"COE_{category}_PREMIUM"] = monthly["PREMIUM"]
        monthly[f"COE_{category}_BID_QUOTA_RATIO"] = monthly["BIDS"] / monthly["QUOTA"].where(monthly["QUOTA"].gt(0))
        coe_parts.append(monthly[[f"COE_{category}_PREMIUM", f"COE_{category}_BID_QUOTA_RATIO"]])
    coe_monthly = pd.concat(coe_parts, axis=1)

    stock = stock.copy()
    stock["DATE"] = pd.to_datetime(stock["DATE"], errors="raise")
    if stock.duplicated(["SYMBOL", "DATE"]).any():
        raise ValueError("Stock SYMBOL/DATE keys must be unique")
    stock["CLOSE"] = pd.to_numeric(stock["CLOSE"], errors="coerce")
    missing_close = int(stock["CLOSE"].isna().sum())
    stock["CLOSE"] = stock["CLOSE"].where(stock["CLOSE"].gt(0))
    stock = stock.sort_values(["SYMBOL", "DATE"])
    stock["_MONTH"] = stock.DATE.dt.to_period("M")
    # No forward/backward fill of missing daily prices.
    stock["_RETURN"] = stock.groupby("SYMBOL")["CLOSE"].pct_change(fill_method=None)
    stock["_RETURN"] = stock["_RETURN"].replace([np.inf, -np.inf], np.nan)
    symbol_month = stock.groupby(["SYMBOL", "_MONTH"]).agg(
        last_close=("CLOSE", "last"), daily_vol=("_RETURN", "std"), observations=("CLOSE", "count"),
    ).reset_index().sort_values(["SYMBOL", "_MONTH"])
    previous_close = symbol_month.groupby("SYMBOL")["last_close"].shift(1)
    previous_month = symbol_month.groupby("SYMBOL")["_MONTH"].shift(1)
    adjacent = symbol_month["_MONTH"].eq(previous_month + 1)
    returns = symbol_month["last_close"] / previous_close - 1
    symbol_month["_MONTH_RETURN"] = returns.where(adjacent & symbol_month.observations.ge(2))
    symbol_month["_POSITIVE"] = symbol_month["_MONTH_RETURN"].gt(0).astype(float).where(symbol_month["_MONTH_RETURN"].notna())
    stock_monthly = symbol_month.groupby("_MONTH").agg(
        STOCK_EQ_WEIGHT_RETURN=("_MONTH_RETURN", "mean"),
        STOCK_MEAN_DAILY_VOL=("daily_vol", "mean"),
        STOCK_POSITIVE_RETURN_SHARE=("_POSITIVE", "mean"),
        STOCK_SYMBOL_COUNT=("_MONTH_RETURN", "count"),
    )
    combined = coe_monthly.join(stock_monthly, how="outer").sort_index()
    combined.index.name = "SOURCE_MONTH"
    audit = {
        "coe_rows": len(coe), "stock_rows": len(stock), "stock_missing_close_rows": missing_close,
        "stock_symbols": int(stock.SYMBOL.nunique()), "monthly_rows": len(combined),
        "coe_rule": "mean premium across rounds; sum bids / sum quota for each month/category",
        "stock_rule": "equal-weight symbol monthly CLOSE returns, mean daily volatility and positive return share; no price fill",
        "lag_months": 2,
    }
    return combined, audit


def attach_macro(df: pd.DataFrame, monthly: pd.DataFrame, cutoff: pd.Timestamp | None = None, lag_months: int = 2) -> pd.DataFrame:
    if lag_months < 1:
        raise ValueError("Macro lag must be at least one month")
    out = df.copy()
    months = pd.to_datetime(out["RENT_APPROVAL_DATE"]).dt.to_period("M")
    source = months - lag_months
    if cutoff is not None:
        # Example: train through Dec; with lag 2, Jan-Mar forecasts all use
        # at most Nov information. February/December validation market values
        # never enter from the supplied auxiliary future rows.
        latest = pd.Timestamp(cutoff).to_period("M") - (lag_months - 1)
        source = source.where(source.le(latest), latest)
    joined = monthly.reindex(pd.PeriodIndex(source))
    joined.index = out.index
    for column in COE_FEATURES + STOCK_FEATURES:
        out[column] = joined[column].to_numpy()
    out["MACRO_SOURCE_MONTH"] = source.astype(str)
    return out
