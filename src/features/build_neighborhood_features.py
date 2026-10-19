"""Causal rental distributions and nearby-block unit-rent summaries.

Exclude the entire observation month for training features. A forecast window
uses one frozen training snapshot, never its own or earlier query labels.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

DATE="RENT_APPROVAL_DATE"
TARGET="MONTHLY_RENT"
AREA="FLOOR_AREA_SQM"
EXACT=["TOWN","BLOCK","STREET","FLAT_TYPE","FLAT_MODEL",AREA,"LEASE_COMMENCE_DATE"]
BLOCK=["TOWN","BLOCK","STREET","FLAT_TYPE"]
LOOKUP_COLUMNS=list(dict.fromkeys([DATE,*EXACT,"LATITUDE","LONGITUDE"]))
DISTRIBUTION=[f"PAST_{group}_{stat}" for group in ["EXACT","BLOCK"] for stat in ["COUNT","MEAN_RATE","MEDIAN_RATE","STD_RATE","Q10_RATE","Q90_RATE"]]
SPATIAL=[f"NEAR_{k}_{stat}" for k in [16,64] for stat in ["BLOCKS","MEAN_RATE","STD_RATE","MEDIAN_RATE","MEAN_DISTANCE_KM","FARTHEST_KM"]]
ALL_FEATURES=DISTRIBUTION+SPATIAL


def area_values(frame: pd.DataFrame) -> np.ndarray:
    values=pd.to_numeric(frame[AREA],errors="raise").to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values<=0).any():
        raise ValueError("Unit-rent modelling requires positive finite floor areas")
    return values


def snapshot(past: pd.DataFrame, query: pd.DataFrame) -> pd.DataFrame:
    """Compute one snapshot; the query target is not read or returned."""
    out=pd.DataFrame(np.nan,index=query.index,columns=ALL_FEATURES)
    out[[c for c in ALL_FEATURES if c.endswith("COUNT") or c.endswith("BLOCKS")]]=0.
    if past.empty or query.empty:
        return out
    if pd.to_datetime(past[DATE]).max()>=pd.to_datetime(query[DATE]).min():
        raise ValueError("Snapshot labels must precede every query month")
    past=past.copy()
    past["_RATE"]=past[TARGET].to_numpy(dtype=float)/area_values(past)
    if not np.isfinite(past._RATE).all():
        raise ValueError("Historical rates must be finite")
    for name,columns in [("EXACT",EXACT),("BLOCK",BLOCK)]:
        groups=past.groupby(columns,dropna=False,observed=True)._RATE
        table=groups.agg(["count","mean","median","std"])
        table["q10"],table["q90"]=groups.quantile(.1),groups.quantile(.9)
        joined=table.reindex(pd.MultiIndex.from_frame(query[columns]))
        for source,stat in [("count","COUNT"),("mean","MEAN_RATE"),("median","MEDIAN_RATE"),("std","STD_RATE"),("q10","Q10_RATE"),("q90","Q90_RATE")]:
            values=joined[source].to_numpy(dtype=float)
            out[f"PAST_{name}_{stat}"]=np.nan_to_num(values,nan=0.) if source=="count" else values
    # Aggregate transaction duplicates into one block/type first. Thus dense
    # blocks cannot fill all nearest-neighbour slots with repeat transactions.
    blocks=past.groupby(BLOCK,dropna=False,observed=True).agg(
        LATITUDE=("LATITUDE","first"),LONGITUDE=("LONGITUDE","first"),rate=("_RATE","mean")).reset_index()
    good=blocks.LATITUDE.between(-90,90)&blocks.LONGITUDE.between(-180,180)
    blocks=blocks.loc[good]
    qgood=query.LATITUDE.between(-90,90)&query.LONGITUDE.between(-180,180)
    for kind in query.FLAT_TYPE.unique():
        reference=blocks.loc[blocks.FLAT_TYPE.eq(kind)]
        mask=query.FLAT_TYPE.eq(kind)&qgood
        if reference.empty or not mask.any():
            continue
        tree=BallTree(np.radians(reference[["LATITUDE","LONGITUDE"]].to_numpy()),metric="haversine")
        distance,indices=tree.query(np.radians(query.loc[mask,["LATITUDE","LONGITUDE"]].to_numpy()),k=min(64,len(reference)))
        rates=reference.rate.to_numpy()[indices]
        for k in [16,64]:
            n=min(k,len(reference))
            values=rates[:,:n]
            out.loc[mask,f"NEAR_{k}_BLOCKS"]=n
            out.loc[mask,f"NEAR_{k}_MEAN_RATE"]=values.mean(axis=1)
            out.loc[mask,f"NEAR_{k}_STD_RATE"]=values.std(axis=1)
            out.loc[mask,f"NEAR_{k}_MEDIAN_RATE"]=np.median(values,axis=1)
            out.loc[mask,f"NEAR_{k}_MEAN_DISTANCE_KM"]=distance[:,:n].mean(axis=1)*6371.0088
            out.loc[mask,f"NEAR_{k}_FARTHEST_KM"]=distance[:,n-1]*6371.0088
    return out


def _cached_snapshot(past: pd.DataFrame, query: pd.DataFrame, cache: Path | None) -> pd.DataFrame:
    if cache is None:
        return snapshot(past,query)
    fingerprint=hashlib.sha256()
    for data,columns in [(past,LOOKUP_COLUMNS+[TARGET]),(query,LOOKUP_COLUMNS)]:
        fingerprint.update(pd.util.hash_pandas_object(data[columns],index=True).to_numpy().tobytes())
    path=cache/f"snapshot_{fingerprint.hexdigest()[:24]}.npz"
    if path.exists():
        with np.load(path) as saved:
            np.testing.assert_array_equal(saved["source_row"],query.index)
            return pd.DataFrame(saved["features"],index=query.index,columns=ALL_FEATURES)
    result=snapshot(past,query)
    cache.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,source_row=query.index.to_numpy(),features=result.to_numpy(dtype=float))
    return result


def causal_features(fit: pd.DataFrame, query: pd.DataFrame, cache: Path | None=None, window: int=12):
    if fit.empty or window<1:
        raise ValueError("Nonempty fit and positive window required")
    fit_month=pd.to_datetime(fit[DATE]).dt.to_period("M")
    query_month=pd.to_datetime(query[DATE]).dt.to_period("M")
    if len(query) and query_month.min()<=fit_month.max():
        raise ValueError("Forecast dates must be strictly after training dates")
    left=fit.copy()
    right=query.drop(columns=TARGET,errors="ignore").copy()
    parts=[]
    for month in sorted(fit_month.unique()):
        past=fit.loc[fit_month.ge(month-window)&fit_month.lt(month)]
        rows=fit.loc[fit_month.eq(month)].drop(columns=TARGET)
        parts.append(_cached_snapshot(past,rows,cache))
    features=pd.concat(parts).reindex(fit.index)
    left[ALL_FEATURES]=features[ALL_FEATURES]
    origin=fit_month.max()+1
    past=fit.loc[fit_month.ge(origin-window)]
    right[ALL_FEATURES]=_cached_snapshot(past,right,cache)[ALL_FEATURES]
    return left,right
