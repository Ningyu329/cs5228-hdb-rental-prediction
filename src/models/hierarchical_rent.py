"""Joint, partially pooled spatial effects and damped calendar trends.

Ridge is a crossed-effects approximation, not a Bayesian posterior model.
All encoders, effects, knots and scaling are learned from fit rows only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Ridge

DATE = "RENT_APPROVAL_DATE"
TARGET = "MONTHLY_RENT"


def month_values(frame):
    dates = pd.to_datetime(frame[DATE])
    return (dates.dt.year * 12 + dates.dt.month).to_numpy(float)


def space_keys(frame):
    out = frame[["TOWN", "FLAT_TYPE", "FLAT_MODEL"]].fillna("missing").astype(str).copy()
    out["TOWN_TYPE"] = out.TOWN + "|" + out.FLAT_TYPE
    out["STREET_TYPE"] = frame.STREET.astype(str) + "|" + out.TOWN_TYPE
    out["BLOCK_TYPE"] = frame.BLOCK.astype(str) + "|" + out.STREET_TYPE
    return out


class HierarchicalRent:
    def __init__(self, alpha=30., damping=6.):
        self.alpha, self.damping = alpha, damping

    def matrix(self, frame):
        space = self.space.transform(space_keys(frame))
        group = self.group.transform(space_keys(frame)[["TOWN_TYPE"]])
        t = month_values(frame) - self.origin
        # Training is unmodified; extrapolated slope progressively flattens.
        effective = np.where(t > 0, self.damping * (-np.expm1(-np.maximum(t, 0) / self.damping)), t)
        calendar = np.column_stack([effective / 12, *[np.maximum(effective + k, 0) / 12 for k in (6, 12, 24)],
                                    np.sin(2 * np.pi * month_values(frame) / 12),
                                    np.cos(2 * np.pi * month_values(frame) / 12)])
        # Region-specific recent slopes are strongly shrunk toward the shared trend.
        recent = (np.maximum(effective, -24) + 24) / 12
        slopes = group.multiply(recent[:, None]) * .25
        num = frame[["FLOOR_AREA_SQM", "LEASE_COMMENCE_DATE", "LATITUDE", "LONGITUDE"]].to_numpy(float)
        num = np.where(np.isfinite(num), num, self.medians)
        num = (num - self.means) / self.scales
        return sparse.hstack([space, sparse.csr_matrix(num), sparse.csr_matrix(calendar), slopes], format="csr")

    def fit(self, frame):
        if frame.empty or not np.isfinite(frame[TARGET]).all():
            raise ValueError("Finite nonempty training labels required")
        self.origin = month_values(frame).max()
        self.space = OneHotEncoder(handle_unknown="ignore", sparse_output=True).fit(space_keys(frame))
        self.group = OneHotEncoder(handle_unknown="ignore", sparse_output=True).fit(space_keys(frame)[["TOWN_TYPE"]])
        num = frame[["FLOOR_AREA_SQM", "LEASE_COMMENCE_DATE", "LATITUDE", "LONGITUDE"]].to_numpy(float)
        self.medians = np.nanmedian(np.where(np.isfinite(num), num, np.nan), axis=0)
        if not np.isfinite(self.medians).all():
            raise ValueError("Entirely missing numeric field")
        num = np.where(np.isfinite(num), num, self.medians)
        self.means, self.scales = num.mean(axis=0), num.std(axis=0)
        self.scales[self.scales == 0] = 1.
        self.model = Ridge(alpha=self.alpha, solver="lsqr", tol=1e-6).fit(self.matrix(frame), frame[TARGET])
        return self

    def predict(self, frame):
        return self.model.predict(self.matrix(frame))
