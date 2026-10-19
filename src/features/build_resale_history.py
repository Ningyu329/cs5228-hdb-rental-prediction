"""Origin-frozen historic resale features, never current/future rental labels."""
from __future__ import annotations

import numpy as np
import pandas as pd
from src.data.clean_data import normalise_text, normalise_flat_type

DATE = "RENT_APPROVAL_DATE"
LEVELS = {"BLOCK": ["TOWN", "STREET", "BLOCK", "FLAT_TYPE"],
          "STREET": ["TOWN", "STREET", "FLAT_TYPE"], "TOWN": ["TOWN", "FLAT_TYPE"]}
STATS = ["COUNT", "MEDIAN_PSM", "AREA_MEAN", "AREA_STD", "HIGH_FLOOR_SHARE"]
FEATURES = [f"RESALE_{level}_{stat}" for level in LEVELS for stat in STATS] + ["RESALE_STALENESS_MONTHS"]

# Exact token substitutions, not fuzzy matching. Both datasets use this key;
# normalized competition STREET itself is never overwritten.
STREET_TOKENS = dict(avenue="ave", road="rd", street="st", drive="dr", crescent="cres",
    central="ctrl", heights="hts", jalan="jln", link="lk", close="cl", terrace="ter",
    north="nth", south="sth", bukit="bt", upper="upp", park="pk", gardens="gdns",
    grove="gr", tanjong="tg", kampong="kg", lorong="lor", mount="mt", place="pl",
    commonwealth="c'wealth", saint="st", **{"st.": "st"})


def canonical_street(value):
    return " ".join(STREET_TOKENS.get(t, t) for t in normalise_text(str(value)).split())


def canonical_town(value):
    text = normalise_text(str(value))
    return "central" if text == "central area" else text


def load_resale(path):
    data = pd.read_csv(path, dtype=str).rename(columns={
        "month": DATE, "town": "TOWN", "block": "BLOCK", "street_name": "STREET",
        "flat_type": "FLAT_TYPE", "floor_area_sqm": "AREA", "resale_price": "PRICE"})
    for c in ["TOWN", "BLOCK", "STREET"]:
        data[c] = data[c].map(normalise_text)
    data["TOWN"] = data.TOWN.map(canonical_town)
    data["STREET"] = data.STREET.map(canonical_street)
    data["FLAT_TYPE"] = data.FLAT_TYPE.map(normalise_flat_type)
    data[DATE] = pd.to_datetime(data[DATE], format="%Y-%m", errors="raise")
    data["MONTH"] = data[DATE].dt.to_period("M").astype("int64")
    data["AREA"] = pd.to_numeric(data.AREA, errors="raise")
    data["PRICE"] = pd.to_numeric(data.PRICE, errors="raise")
    if not ((data.AREA > 0) & (data.PRICE > 0)).all():
        raise ValueError("Invalid resale area/price")
    data["PSM"] = data.PRICE / data.AREA
    lower = pd.to_numeric(data.storey_range.str.extract(r"^(\d+)")[0], errors="raise")
    data["HIGH_FLOOR"] = lower.ge(10).astype(float)
    return data


def snapshot(history, query, cutoff, window=24):
    """cutoff is an inclusive maximum public event month; never infer it from history."""
    past = history.loc[history.MONTH.between(cutoff - window + 1, cutoff)]
    out = pd.DataFrame(index=query.index)
    query_keys = query.copy()
    query_keys["TOWN"] = query.TOWN.map(canonical_town)
    query_keys["STREET"] = query.STREET.map(canonical_street)
    for level, keys in LEVELS.items():
        group = past.groupby(keys, observed=True).agg(COUNT=("PSM", "size"),
            MEDIAN_PSM=("PSM", "median"), AREA_MEAN=("AREA", "mean"),
            AREA_STD=("AREA", "std"), HIGH_FLOOR_SHARE=("HIGH_FLOOR", "mean"))
        merged = query_keys[keys].join(group, on=keys, validate="many_to_one")
        for stat in STATS:
            out[f"RESALE_{level}_{stat}"] = merged[stat].to_numpy()
        out[f"RESALE_{level}_COUNT"] = out[f"RESALE_{level}_COUNT"].fillna(0)
    out["RESALE_STALENESS_MONTHS"] = query[DATE].dt.to_period("M").astype("int64") - cutoff
    return out


def causal_resale(fit, query, history, lag=2, window=24):
    if lag < 1 or window < 1 or fit.empty:
        raise ValueError("Strictly positive lag/window and nonempty fit required")
    fit_month = fit[DATE].dt.to_period("M").astype("int64")
    origin = int(fit_month.max())
    if not query.empty and (query[DATE].dt.to_period("M").astype("int64") <= origin).any():
        raise ValueError("Query must start after complete training months")
    train_features = pd.concat([snapshot(history, fit.loc[fit_month.eq(m)], int(m) - lag, window)
                                for m in sorted(fit_month.unique())]).reindex(fit.index)
    future = snapshot(history, query, origin - lag, window)
    a, b = fit.copy(), query.drop(columns="MONTHLY_RENT", errors="ignore").copy()
    a[FEATURES], b[FEATURES] = train_features[FEATURES], future[FEATURES]
    return a, b
