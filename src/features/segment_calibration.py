"""Segment diagnostics and shrunken corrections fitted on past OOF errors."""
from __future__ import annotations

import numpy as np
import pandas as pd

DATE = "RENT_APPROVAL_DATE"
TARGET = "MONTHLY_RENT"
PARTITIONS = {
    "flat": ["FLAT_TYPE"], "town": ["TOWN"], "region": ["REGION"],
    "month": [DATE], "area": ["AREA_BIN"], "age": ["AGE_BIN"],
    "town_flat": ["TOWN", "FLAT_TYPE"],
    "region_flat": ["REGION", "FLAT_TYPE"],
    "flat_month": ["FLAT_TYPE", DATE],
    "flat_age": ["FLAT_TYPE", "AGE_BIN"],
    "area_age": ["AREA_BIN", "AGE_BIN"],
    "address_history": ["ADDRESS_HISTORY_BIN"],
}


def month_index(frame: pd.DataFrame) -> np.ndarray:
    date = pd.to_datetime(frame[DATE])
    return (date.dt.year * 12 + date.dt.month).to_numpy(dtype=int)


def keys_for(frame: pd.DataFrame, columns: list[str]) -> list[tuple[str, ...]]:
    if not columns:
        return [()] * len(frame)
    return list(frame[columns].astype("string").fillna("missing").itertuples(index=False, name=None))


def segments(frame: pd.DataFrame, prediction: np.ndarray, fit: pd.DataFrame) -> pd.DataFrame:
    """Every partition covers every row once; different partitions overlap."""
    view = frame.copy()
    prediction = np.asarray(prediction, dtype=float)
    if len(view) != len(prediction) or not np.isfinite(prediction).all():
        raise ValueError("Invalid predictions")
    view[DATE] = pd.to_datetime(view[DATE]).dt.strftime("%Y-%m")
    view["AREA_BIN"] = pd.cut(view.FLOOR_AREA_SQM, [-np.inf, 50, 70, 90, 110, 130, 160, np.inf]).astype(str)
    view["AGE_BIN"] = pd.cut(view.FLAT_AGE, [-np.inf, 10, 20, 30, 40, 50, np.inf]).astype("string").fillna("missing")
    counts = fit.BLOCK_ADDRESS.value_counts()
    n = view.BLOCK_ADDRESS.map(counts).fillna(0)
    view["ADDRESS_HISTORY_BIN"] = pd.cut(n, [-1, 0, 4, 19, np.inf], labels=["unseen", "1-4", "5-19", "20+"]).astype(str)
    view["_ERROR"] = prediction - view[TARGET].to_numpy()
    view["_SSE"] = view._ERROR ** 2
    total = float(view._SSE.sum())
    rows = []
    for partition, columns in PARTITIONS.items():
        grouper = columns[0] if len(columns) == 1 else columns
        for value, group in view.groupby(grouper, dropna=False, observed=True):
            values = value if isinstance(value, tuple) else (value,)
            error = group._ERROR.to_numpy()
            sse = float(np.dot(error, error))
            rows.append(dict(partition=partition, segment=" / ".join(map(str, values)),
                rows=len(group), row_share=len(group) / len(view), rmse=float(np.sqrt(np.mean(error ** 2))),
                mae=float(np.abs(error).mean()), bias=float(error.mean()), sse=sse,
                sse_share=sse / total if total else 0.0,
                bias_sse_share=len(group) * float(error.mean()) ** 2 / total if total else 0.0,
                abs_error_p90=float(np.quantile(np.abs(error), .9)),
                errors_over_1000=int((np.abs(error) > 1000).sum())))
    return pd.DataFrame(rows).sort_values(["partition", "sse"], ascending=[True, False]).reset_index(drop=True)


def fit_correction(oof: pd.DataFrame, columns: list[str], shrink: float = 100,
                   trend: bool = False, min_rows: int = 30) -> dict:
    """OOF must contain predictions generated before seeing each row's label.

    This routine does not claim to verify upstream model provenance; callers
    must supply expanding-origin OOF rows. Never fit on in-sample residuals.
    """
    if oof.empty or shrink < 0 or min_rows < 1:
        raise ValueError("Need calibration data, nonnegative shrink and positive min_rows")
    months = month_index(oof)
    cutoff = int(months.max())
    residual = oof[TARGET].to_numpy(dtype=float) - oof["prediction"].to_numpy(dtype=float)
    if not np.isfinite(residual).all():
        raise ValueError("Residuals must be finite")
    indices: dict[tuple, list[int]] = {}
    for i, key in enumerate(keys_for(oof, columns)):
        indices.setdefault(key, []).append(i)
    params = {}
    for key, positions in indices.items():
        if len(positions) < min_rows:
            continue
        positions = np.asarray(positions)
        level, slope = float(residual[positions].mean()), 0.0
        if trend:
            data = pd.DataFrame({"month": months[positions], "residual": residual[positions]})
            monthly = data.groupby("month").residual.agg(["mean", "count"])
            if len(monthly) < 4:
                continue
            x = monthly.index.to_numpy() - cutoff
            design = np.column_stack([np.ones(len(x)), x])
            weight = np.sqrt(monthly["count"].to_numpy())
            level, slope = np.linalg.lstsq(design * weight[:, None], monthly["mean"].to_numpy() * weight, rcond=None)[0]
            slope = float(np.clip(slope, -25, 25))
        reliability = len(positions) / (len(positions) + shrink)
        params[key] = (float(level) * reliability, float(slope) * reliability, len(positions))
    return dict(columns=columns, cutoff=cutoff, params=params, trend=trend,
        shrink=shrink, min_rows=min_rows, calibration_rows=len(oof))


def correction_values(model: dict, query: pd.DataFrame) -> np.ndarray:
    """Future query labels are ignored; unseen/small segments receive zero."""
    months = month_index(query)
    if len(months) and months.min() <= model["cutoff"]:
        raise ValueError("Calibration must end strictly before the forecast window")
    out = np.zeros(len(query))
    for i, key in enumerate(keys_for(query, model["columns"])):
        level, slope, _ = model["params"].get(key, (0., 0., 0))
        horizon = months[i] - model["cutoff"]
        out[i] = np.clip(level + slope * 6 * (1 - np.exp(-horizon / 6)), -200, 200)
    return out


def available_macro(frame: pd.DataFrame, monthly: pd.DataFrame) -> pd.DataFrame:
    """Condition on supplied same-month covariates, not a fixed-origin forecast.

    No fill beyond available rows, and no rental targets are consulted. This
    is a different information regime from cutoff-frozen macro experiments.
    """
    out = frame.copy()
    months = pd.to_datetime(frame[DATE]).dt.to_period("M")
    joined = monthly.reindex(pd.PeriodIndex(months))
    for column in monthly.columns:
        out[column] = joined[column].to_numpy()
    out["MACRO_SOURCE_MONTH"] = months.astype(str).to_numpy()
    return out
