"""Out-of-fold target-encoding LightGBM experiments on the standard time windows.

Fast, self-contained screening: builds K-fold target encodings of level-adjusted
rent for fine address/flat groups (training rows get out-of-fold values, query
rows use the whole training window), fits LightGBM, and compares/blends against
the stored incumbent predictions. Does not read the competition test set unless
``--export`` is given.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TRAIN = ROOT / "data/processed/train_location_features.csv"
TEST = ROOT / "data/processed/test_location_features.csv"
INCUMBENT = ROOT / "reports/experiments/keras_v1/predictions_{split}.csv"

WINDOWS = {
    "q3_2024": ("2024-07", "2024-09"),
    "q4_2024": ("2024-10", "2024-12"),
    "primary_2025": ("2025-01", "2025-03"),
    "long_12m": ("2024-01", "2024-12"),
}
CATS = ["TOWN", "FLAT_TYPE", "FLAT_MODEL", "REGION", "PLANNING_AREA", "SUBZONE"]
NUMS = [
    "FLOOR_AREA_SQM", "LEASE_COMMENCE_DATE", "FLAT_AGE", "LATITUDE", "LONGITUDE",
    "YEAR_COMPLETED", "MAX_FLOOR", "NEAREST_OPEN_MRT_DISTANCE_KM", "OPEN_MRT_COUNT_WITHIN_500M",
    "OPEN_MRT_COUNT_WITHIN_1000M", "NEAREST_SCHOOL_DISTANCE_KM", "SCHOOL_COUNT_WITHIN_1000M",
    "NEAREST_MALL_DISTANCE_KM", "MALL_COUNT_WITHIN_1000M", "BUILDING_AGE", "RENT_MONTH",
]
TE_GROUPS = {
    "addr_ft": ["ADDRESS_KEY", "FLAT_TYPE"],
    "addr_ft_area": ["ADDRESS_KEY", "FLAT_TYPE", "FLOOR_AREA_SQM"],
    "addr": ["ADDRESS_KEY"],
    "street_ft": ["STREET", "FLAT_TYPE"],
    "subzone_ft": ["SUBZONE", "FLAT_TYPE"],
    "town_ft_model": ["TOWN", "FLAT_TYPE", "FLAT_MODEL"],
}
RECENT_GROUPS = {"addr_ft": ["ADDRESS_KEY", "FLAT_TYPE"], "street_ft": ["STREET", "FLAT_TYPE"]}
SMOOTH = 5.0
OPTIONS = {"recent_months": 0, "resale": None, "property": None, "model": "lightgbm", "level": "flat3", "stats": False, "knn": False, "target": "ratio", "volume": False}
PROPERTY_PATH = ROOT / "data/raw/external/hdb_20261019/property.csv"


def load_property() -> pd.DataFrame:
    """Static HDB block composition (latest public snapshot), keyed by block + canonical street."""
    from src.data.clean_data import normalise_text
    from src.features.build_resale_history import canonical_street
    p = pd.read_csv(PROPERTY_PATH, dtype=str)
    key = p["blk_no"].map(normalise_text) + "|" + p["street"].map(canonical_street)
    units = p.filter(regex="_(sold|rental)$").apply(pd.to_numeric)
    total = pd.to_numeric(p["total_dwelling_units"]).clip(lower=1)
    out = pd.DataFrame({"PROP_UNITS": total}, index=p.index)
    for c in units:
        out[f"PROP_SHARE_{c}"] = units[c] / total
    out["PROP_RENTAL_SHARE"] = units.filter(regex="_rental$").sum(axis=1) / total
    for c in ["commercial", "market_hawker", "multistorey_carpark", "precinct_pavilion", "miscellaneous"]:
        out[f"PROP_{c.upper()}"] = (p[c] == "Y").astype(float)
    out.index = key
    if out.index.duplicated().any():
        raise ValueError("Duplicate property keys")
    return out


def property_features(df: pd.DataFrame) -> pd.DataFrame:
    from src.data.clean_data import normalise_text
    from src.features.build_resale_history import canonical_street
    key = df["BLOCK"].astype(str).map(normalise_text) + "|" + df["STREET"].map(canonical_street)
    return OPTIONS["property"].reindex(key.to_numpy()).set_axis(df.index)
RESALE_COLS = ["RESALE_BLOCK_MEDIAN_PSM", "RESALE_BLOCK_COUNT", "RESALE_STREET_MEDIAN_PSM", "RESALE_TOWN_MEDIAN_PSM"]


def resale_features(df: pd.DataFrame, cutoff: int) -> pd.DataFrame:
    """Static block resale snapshot from public events <= cutoff (origin minus 2 months)."""
    from src.features.build_resale_history import snapshot
    keys = df[["TOWN", "STREET", "BLOCK", "FLAT_TYPE"]].copy()
    keys["RENT_APPROVAL_DATE"] = pd.to_datetime(df["RENT_APPROVAL_DATE"])
    return snapshot(OPTIONS["resale"], keys, cutoff, window=36)[RESALE_COLS]


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["MONTH_ID"] = pd.PeriodIndex(df["RENT_APPROVAL_DATE"], freq="M").asi8
    for c in CATS:
        df[c] = df[c].astype(str)
    return df


def month_level(train: pd.DataFrame) -> pd.Series:
    """Mix-adjusted monthly level: mean of rent ratio to its town/flat-type mean."""
    base = train.groupby(["TOWN", "FLAT_TYPE"])["MONTHLY_RENT"].transform("mean")
    return (train["MONTHLY_RENT"] / base).groupby(train["MONTH_ID"]).mean()


def encode_groups(fit: pd.DataFrame, target: np.ndarray, query: pd.DataFrame, prior: float,
                  last_month: int | None = None) -> pd.DataFrame:
    out = {}
    frame = fit.assign(_t=target)
    groups = [(n, c, frame) for n, c in TE_GROUPS.items()]
    if OPTIONS["recent_months"] and last_month is not None:
        recent = frame[frame["MONTH_ID"] > last_month - OPTIONS["recent_months"]]
        groups += [(f"recent_{n}", c, recent) for n, c in RECENT_GROUPS.items()]
    for name, cols, frame in groups:
        stats = frame.groupby(cols)["_t"].agg(["sum", "count"])
        joined = query[cols].merge(stats, left_on=cols, right_index=True, how="left")
        s = joined["sum"].fillna(0.0).to_numpy()
        n = joined["count"].fillna(0.0).to_numpy()
        out[f"te_{name}"] = (s + SMOOTH * prior) / (n + SMOOTH)
        out[f"cnt_{name}"] = n
    if OPTIONS["stats"]:
        for name in ["addr_ft", "street_ft"]:
            cols = TE_GROUPS[name]
            stats = frame.groupby(cols)["_t"].agg(["median", "std"])
            joined = query[cols].merge(stats, left_on=cols, right_index=True, how="left")
            out[f"med_{name}"] = joined["median"].to_numpy()
            out[f"std_{name}"] = joined["std"].to_numpy()
    out = pd.DataFrame(out, index=query.index)
    if OPTIONS["knn"]:
        out = pd.concat([out, knn_features(fit, target, query)], axis=1)
    return out


VOLUME_KEYS = {
    "vol_addr": ["ADDRESS_KEY"], "vol_addr_ft": ["ADDRESS_KEY", "FLAT_TYPE"],
    "vol_exact": ["ADDRESS_KEY", "FLAT_TYPE", "FLOOR_AREA_SQM", "FLAT_MODEL"],
    "vol_town_ft": ["TOWN", "FLAT_TYPE"], "vol_town": ["TOWN"],
}


def volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Same-month listing counts from the rows' own features (no labels); town counts as month share."""
    month_total = df.groupby("MONTH_ID")["MONTH_ID"].transform("size")
    out = {}
    for name, cols in VOLUME_KEYS.items():
        n = df.groupby(cols + ["MONTH_ID"])["MONTH_ID"].transform("size")
        out[name] = n / month_total if name.startswith("vol_town") else n
    return pd.DataFrame(out, index=df.index)


KM_PER_DEG = np.array([110.57, 111.32 * np.cos(np.radians(1.35))])


def knn_features(fit: pd.DataFrame, target: np.ndarray, query: pd.DataFrame, k: int = 8,
                 radius_km: float = 0.5) -> pd.DataFrame:
    """Same-flat-type rent level of the k nearest *other* blocks and of blocks within radius."""
    from scipy.spatial import cKDTree
    agg = fit.assign(_t=target).groupby(["ADDRESS_KEY", "FLAT_TYPE"]).agg(
        s=("_t", "sum"), n=("_t", "count"), lat=("LATITUDE", "first"), lon=("LONGITUDE", "first")).reset_index()
    keys = query[["ADDRESS_KEY", "FLAT_TYPE", "LATITUDE", "LONGITUDE"]].drop_duplicates(["ADDRESS_KEY", "FLAT_TYPE"])
    res = []
    for ft, q in keys.groupby("FLAT_TYPE"):
        a = agg[agg["FLAT_TYPE"] == ft].reset_index(drop=True)
        r = pd.DataFrame({"ADDRESS_KEY": q["ADDRESS_KEY"].to_numpy(), "FLAT_TYPE": ft})
        if len(a) < 2:
            res.append(r)
            continue
        tree = cKDTree(a[["lat", "lon"]].to_numpy() * KM_PER_DEG)
        qxy = q[["LATITUDE", "LONGITUDE"]].to_numpy() * KM_PER_DEG
        kk = min(k + 1, len(a))
        dist, idx = tree.query(qxy, k=kk)
        own = a["ADDRESS_KEY"].to_numpy()[idx] == q["ADDRESS_KEY"].to_numpy()[:, None]
        keep = (~own) & (np.cumsum(~own, axis=1) <= k)
        s_arr, n_arr = a["s"].to_numpy()[idx], a["n"].to_numpy()[idx]
        r["knn_te"] = (s_arr * keep).sum(1) / np.maximum((n_arr * keep).sum(1), 1)
        r["knn_dist"] = (dist * keep).sum(1) / np.maximum(keep.sum(1), 1)
        rad_s, rad_n = np.zeros(len(q)), np.zeros(len(q))
        for i, nb in enumerate(tree.query_ball_point(qxy, radius_km)):
            nb = np.asarray(nb, dtype=int)
            nb = nb[a["ADDRESS_KEY"].to_numpy()[nb] != r["ADDRESS_KEY"].iat[i]]
            rad_s[i], rad_n[i] = a["s"].to_numpy()[nb].sum(), a["n"].to_numpy()[nb].sum()
        r["rad_te"] = np.where(rad_n > 0, rad_s / np.maximum(rad_n, 1), np.nan)
        r["rad_cnt"] = rad_n
        res.append(r)
    feats = pd.concat(res)
    return query[["ADDRESS_KEY", "FLAT_TYPE"]].merge(feats, on=["ADDRESS_KEY", "FLAT_TYPE"], how="left") \
        .drop(columns=["ADDRESS_KEY", "FLAT_TYPE"]).set_axis(query.index)


def build_features(train: pd.DataFrame, query: pd.DataFrame, horizon_level: str, seed: int):
    level = month_level(train)
    last = level.iloc[-3:].mean()
    train_level = train["MONTH_ID"].map(level).to_numpy()
    query_level = query["MONTH_ID"].map(level).fillna(last).to_numpy()
    horizon = (query["MONTH_ID"] - train["MONTH_ID"].max()).clip(lower=0).to_numpy()
    if horizon_level == "flat3":
        future = np.full(len(query), last)
    elif horizon_level.startswith("damped"):
        # Slope of the last 12 monthly levels, extrapolated from the flat3 anchor
        # (centred one month back) with per-month damping phi.
        phi = float(horizon_level.removeprefix("damped"))
        recent = level.iloc[-12:]
        slope = np.polyfit(np.arange(len(recent)), recent.to_numpy(), 1)[0]
        steps = horizon + 1
        damp = steps.astype(float) if phi == 1 else phi * (1 - phi ** steps) / (1 - phi)
        future = last + slope * damp
    else:
        raise ValueError(horizon_level)
    query_level = np.where(horizon > 0, future, query_level)
    adj = train["MONTHLY_RENT"].to_numpy() / train_level
    last = int(train["MONTH_ID"].max())
    prior = float(adj.mean())

    oof = pd.DataFrame(index=train.index)
    parts = []
    for fit_idx, enc_idx in KFold(5, shuffle=True, random_state=seed).split(train):
        f = train.iloc[fit_idx]
        parts.append(encode_groups(f, adj[fit_idx], train.iloc[enc_idx], prior, last))
    oof = pd.concat(parts).loc[train.index]
    q_te = encode_groups(train, adj, query, prior, last)

    def assemble(df, te):
        x = df[NUMS].copy()
        for c in CATS:
            x[c] = df[c]
        return pd.concat([x, te], axis=1)

    if OPTIONS["resale"] is not None:
        oof = pd.concat([oof, resale_features(train, last - 2)], axis=1)
        q_te = pd.concat([q_te, resale_features(query, last - 2)], axis=1)
    if OPTIONS["volume"]:
        oof = pd.concat([oof, volume_features(train)], axis=1)
        q_te = pd.concat([q_te, volume_features(query)], axis=1)
    if OPTIONS["property"] is not None:
        oof = pd.concat([oof, property_features(train)], axis=1)
        q_te = pd.concat([q_te, property_features(query)], axis=1)
    x_tr, x_q = assemble(train, oof), assemble(query, q_te)
    for c in CATS:
        cats = pd.CategoricalDtype(sorted(train[c].unique()))
        x_tr[c] = x_tr[c].astype(cats)
        x_q[c] = x_q[c].astype(cats)
    return x_tr, adj, train_level, x_q, query_level


PARAMS = dict(
    objective="regression", learning_rate=0.03, num_leaves=127, min_data_in_leaf=40,
    feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0,
    cat_smooth=20, cat_l2=10, max_cat_to_onehot=8, verbose=-1,
)


CB_PARAMS = dict(loss_function="RMSE", learning_rate=0.06, depth=8, l2_leaf_reg=5.0,
                 one_hot_max_size=8, verbose=0)
CB_EXTRA_CATS = ["ADDRESS_KEY", "STREET"]


def to_catboost(x: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    x = x.copy()
    for c in CATS:
        x[c] = x[c].astype(str)
    for c in CB_EXTRA_CATS:
        x[c] = df[c].astype(str).to_numpy()
    return x


def train_model(x_tr, y, seed, rounds, threads, valid=None):
    """Returns (model, best_iteration). ``valid`` = (x, y) enables early stopping."""
    if OPTIONS["model"] == "catboost":
        from catboost import CatBoostRegressor
        model = CatBoostRegressor(**CB_PARAMS, iterations=rounds, random_seed=seed, thread_count=threads,
                                  cat_features=CATS + CB_EXTRA_CATS)
        if valid is None:
            model.fit(x_tr, y)
            return model, rounds
        model.fit(x_tr, y, eval_set=valid, early_stopping_rounds=200, use_best_model=True)
        return model, model.get_best_iteration() + 1
    params = dict(PARAMS, seed=seed, num_threads=threads)
    if valid is None:
        return lgb.train(params, lgb.Dataset(x_tr, y), num_boost_round=rounds), rounds
    booster = lgb.train(params, lgb.Dataset(x_tr, y), num_boost_round=rounds,
                        valid_sets=[lgb.Dataset(valid[0], valid[1])],
                        callbacks=[lgb.early_stopping(200, verbose=False)])
    return booster, booster.best_iteration


def model_inputs(train, query, seed):
    x_tr, adj, tr_lvl, x_q, q_lvl = build_features(train, query, OPTIONS["level"], seed)
    if OPTIONS["model"] == "catboost":
        x_tr, x_q = to_catboost(x_tr, train), to_catboost(x_q, query)
    return x_tr, adj, x_q, q_lvl


def fit_predict(train: pd.DataFrame, query: pd.DataFrame, seeds: list[int], rounds: int, threads: int):
    preds = []
    for seed in seeds:
        x_tr, adj, x_q, q_lvl = model_inputs(train, query, seed)
        if OPTIONS["target"] == "log":
            model, _ = train_model(x_tr, np.log(adj), seed, rounds, threads)
            # Smearing correction so the training mean ratio is preserved.
            scale = adj.mean() / np.exp(model.predict(x_tr)).mean()
            preds.append(np.exp(model.predict(x_q)) * scale * q_lvl)
        else:
            model, _ = train_model(x_tr, adj, seed, rounds, threads)
            preds.append(model.predict(x_q) * q_lvl)
    return np.mean(preds, axis=0), model


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def choose_rounds(train: pd.DataFrame, threads: int, cap: int = 4000) -> int:
    """Pick boosting rounds on the last 3 training months (inner time split)."""
    months = sorted(train["MONTH_ID"].unique())
    inner = train["MONTH_ID"] >= months[-3]
    tr, va = train[~inner], train[inner]
    x_tr, adj, x_va, va_lvl = model_inputs(tr, va, 0)
    va_target = va["MONTHLY_RENT"].to_numpy() / va_lvl
    if OPTIONS["target"] == "log":
        adj, va_target = np.log(adj), np.log(va_target)
    _, best = train_model(x_tr, adj, 0, cap, threads, valid=(x_va, va_target))
    return int(best * len(train) / len(tr))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=list(WINDOWS))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=0, help="0 = inner early stopping")
    ap.add_argument("--out", default=str(ROOT / "reports/experiments/te_v1"))
    ap.add_argument("--recent-months", type=int, default=0)
    ap.add_argument("--lr", type=float, default=PARAMS["learning_rate"])
    ap.add_argument("--resale", action="store_true", help="add lagged public resale block features")
    ap.add_argument("--property", action="store_true", help="add static HDB block composition features")
    ap.add_argument("--model", choices=["lightgbm", "catboost"], default="lightgbm")
    ap.add_argument("--level", default="flat3", help="flat3 or damped<phi>, e.g. damped0.9")
    ap.add_argument("--stats", action="store_true", help="add group median/std encodings")
    ap.add_argument("--knn", action="store_true", help="add spatial nearest-block encodings")
    ap.add_argument("--params", default="{}", help="JSON overrides for LightGBM params")
    ap.add_argument("--target", choices=["ratio", "log"], default="ratio")
    ap.add_argument("--volume", action="store_true", help="add same-month listing-count features")
    ap.add_argument("--export", action="store_true", help="fit on all rows and predict test")
    args = ap.parse_args()
    out = Path(args.out)
    OPTIONS["recent_months"] = args.recent_months
    PARAMS["learning_rate"] = args.lr
    PARAMS.update(json.loads(args.params))
    OPTIONS["target"] = args.target
    OPTIONS["volume"] = args.volume
    OPTIONS["model"] = args.model
    OPTIONS["level"] = args.level
    OPTIONS["stats"], OPTIONS["knn"] = args.stats, args.knn
    if args.property:
        OPTIONS["property"] = load_property()
    if args.resale:
        from src.features.build_resale_history import load_resale
        OPTIONS["resale"] = load_resale(ROOT / "data/raw/external/hdb_20261019/resale.csv")
    out.mkdir(parents=True, exist_ok=True)
    data = load(TRAIN)
    seeds = list(range(args.seeds))

    if args.export:
        rounds = args.rounds or choose_rounds(data, args.threads)
        test = load(TEST)
        pred, _ = fit_predict(data, test, seeds, rounds, args.threads)
        pd.DataFrame({"Id": np.arange(len(test)), "Predicted": pred}).to_csv(out / "te_test_predictions.csv", index=False)
        print(json.dumps({"rounds": rounds, "mean": float(pred.mean())}))
        return

    rows = []
    for split in args.splits:
        start, end = WINDOWS[split]
        s, e = pd.Period(start, "M").ordinal, pd.Period(end, "M").ordinal
        train = data[data["MONTH_ID"] < s]
        query = data[(data["MONTH_ID"] >= s) & (data["MONTH_ID"] <= e)]
        rounds = args.rounds or choose_rounds(train, args.threads)
        pred, booster = fit_predict(train, query, seeds, rounds, args.threads)
        inc = pd.read_csv(str(INCUMBENT).format(split=split)).set_index("source_row").loc[query.index]
        y = query["MONTHLY_RENT"].to_numpy()
        assert np.allclose(inc["actual"].to_numpy(), y)
        pd.DataFrame({"source_row": query.index, "actual": y, "te": pred, "incumbent": inc["incumbent"].to_numpy()}).to_csv(
            out / f"predictions_{split}.csv", index=False)
        row = {"split": split, "rows": len(y), "rounds": rounds, "incumbent": rmse(inc["incumbent"], y), "te": rmse(pred, y),
               "te_bias": float(np.mean(pred - y)),
               "resid_corr": float(np.corrcoef(pred - y, inc["incumbent"] - y)[0, 1])}
        for w in (0.25, 0.5, 0.75):
            row[f"blend{int(w * 100)}"] = rmse(w * pred + (1 - w) * inc["incumbent"], y)
        rows.append(row)
        print(json.dumps(row), flush=True)
        if split == args.splits[0] and args.model == "lightgbm":
            imp = pd.Series(booster.feature_importance("gain"), index=booster.feature_name()).sort_values(ascending=False)
            imp.to_csv(out / f"importance_{split}.csv")
    pd.DataFrame(rows).to_csv(out / "results.csv", index=False)


if __name__ == "__main__":
    main()
