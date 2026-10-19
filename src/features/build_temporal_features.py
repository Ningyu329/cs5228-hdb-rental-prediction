"""Past-only rent statistics and explicit, bounded monthly trend models.

Training statistics exclude the entire observation month. A forecast block is
frozen at the last training month, even if its query dates span many months.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "MONTHLY_RENT"
DATE = "RENT_APPROVAL_DATE"
GROUPS = {
    "TOWN_FLAT": ["TOWN", "FLAT_TYPE"],
    "STREET_FLAT": ["TOWN", "STREET", "FLAT_TYPE"],
    "BLOCK_FLAT": ["TOWN", "STREET", "BLOCK", "FLAT_TYPE"],
}


def month_number(frame: pd.DataFrame) -> pd.Series:
    dates = pd.to_datetime(frame[DATE])
    return dates.dt.year * 12 + dates.dt.month


def history_columns(window: int) -> list[str]:
    return [f"HIST_{name}_{window}M_{stat}" for name in GROUPS for stat in ["MEAN", "COUNT"]]


def historical_features(
    fit: pd.DataFrame, query: pd.DataFrame, window: int = 6, smoothing: float = 20.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create expanding-origin training features and frozen query features.

    Group means shrink toward the same past window's global mean. A new group
    gets that past global mean and count zero. With no history, mean is NaN;
    no future/global-all-data target average is used as a fallback.
    """
    if window < 1 or smoothing <= 0 or fit.empty:
        raise ValueError("Need nonempty fit, positive history window and smoothing")
    fit_month = month_number(fit)
    query_month = month_number(query)
    if len(query) and query_month.min() <= fit_month.max():
        raise ValueError("Forecast query must be strictly after all training months")
    left, right = fit.copy(), query.copy()
    columns = history_columns(window)
    for frame in [left, right]:
        for column in columns:
            frame[column] = 0.0 if column.endswith("COUNT") else np.nan
    for forecast_month in sorted(fit_month.unique()) + [int(fit_month.max()) + 1]:
        past = fit.loc[fit_month.ge(forecast_month - window) & fit_month.lt(forecast_month)]
        training = forecast_month <= fit_month.max()
        dest = left if training else right
        mask = fit_month.eq(forecast_month) if training else pd.Series(True, index=query.index)
        if not mask.any() or past.empty:
            continue
        prior = float(past[TARGET].mean())
        for name, keys in GROUPS.items():
            grouped = past.groupby(keys, dropna=False)[TARGET].agg(["sum", "count"])
            lookup = pd.MultiIndex.from_frame(dest.loc[mask, keys])
            counts = grouped["count"].reindex(lookup).fillna(0).to_numpy(dtype=float)
            sums = grouped["sum"].reindex(lookup).fillna(0).to_numpy(dtype=float)
            dest.loc[mask, f"HIST_{name}_{window}M_MEAN"] = (sums + smoothing * prior) / (counts + smoothing)
            dest.loc[mask, f"HIST_{name}_{window}M_COUNT"] = counts
    return left, right


def fit_trend(fit: pd.DataFrame, kind: str) -> dict:
    """Fit a monthly level plus a predeclared future trend from fit labels only.

    Observed training months use their own global level (target decomposition,
    not a feature). Only static housing features predict the remaining residual.
    Forecast slope is damped with a six-month time constant. Monthly averages
    can reflect changing housing composition, a limitation checked by backtests.
    """
    months = month_number(fit)
    monthly = fit.assign(_MONTH=months).groupby("_MONTH")[TARGET].agg(["mean", "count"])
    end = int(monthly.index.max())
    if kind == "flat3":
        tail = monthly.tail(3)
        level = float(np.average(tail["mean"], weights=tail["count"]))
        slope = 0.0
    elif kind in {"damped12", "damped24"}:
        width = int(kind.removeprefix("damped"))
        tail = monthly.loc[monthly.index > end - width]
        if len(tail) < 3:
            raise ValueError("Trend needs at least three distinct months")
        x = tail.index.to_numpy(dtype=float) - end
        design = np.column_stack([np.ones(len(x)), x])
        weight = np.sqrt(tail["count"].to_numpy(dtype=float))
        level, slope = np.linalg.lstsq(design * weight[:, None], tail["mean"].to_numpy() * weight, rcond=None)[0]
        level, slope = float(level), float(slope)
    else:
        raise ValueError(f"Unknown trend: {kind}")
    return dict(kind=kind, cutoff=end, level=level, slope=slope, damping_months=6,
        monthly_levels={int(month): float(row["mean"]) for month, row in monthly.iterrows()})


def trend_values(trend: dict, frame: pd.DataFrame) -> np.ndarray:
    months = month_number(frame).to_numpy(dtype=int)
    horizon = np.maximum(months - trend["cutoff"], 0)
    values = trend["level"] + trend["slope"] * trend["damping_months"] * (1 - np.exp(-horizon / trend["damping_months"]))
    for index, month in enumerate(months):
        if month <= trend["cutoff"]:
            if month not in trend["monthly_levels"]:
                raise ValueError("Training trend month is absent")
            values[index] = trend["monthly_levels"][month]
    return values.astype(float)
