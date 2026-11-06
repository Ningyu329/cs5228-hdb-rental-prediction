"""Train simple and supervised baselines with time/random validation.

Run from the project root:

    python src/models/train_baseline.py

The primary validation split is the last three months in the labelled
training data.  A random 80/20 split is also reported as a secondary
reference.  If CatBoost is installed, ``--model auto`` uses it.  Otherwise the
script uses a reproducible one-hot Ridge fallback so the first experiment can
still be completed without an extra dependency.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
REPORT_DIR = PROJECT_ROOT / "reports" / "experiments"
SUBMISSION_DIR = PROJECT_ROOT / "submissions"

TARGET = "MONTHLY_RENT"
CATEGORICAL_FEATURES = ["TOWN", "BLOCK", "STREET", "FLAT_TYPE", "FLAT_MODEL"]
NUMERIC_FEATURES = [
    "FLOOR_AREA_SQM",
    "LEASE_COMMENCE_DATE",
    "RENT_YEAR",
    "RENT_MONTH",
    "RENT_MONTH_INDEX",
    "FLAT_AGE",
]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(INTERIM_DIR / "train_clean.csv")
    test = pd.read_csv(INTERIM_DIR / "test_clean.csv")
    for df in [train, test]:
        for column in NUMERIC_FEATURES:
            df[column] = pd.to_numeric(df[column], errors="coerce")
        for column in CATEGORICAL_FEATURES:
            df[column] = df[column].fillna("missing").astype(str)
    train = train.dropna(subset=[TARGET]).copy()
    train["RENT_APPROVAL_DATE"] = pd.to_datetime(train["RENT_APPROVAL_DATE"])
    test["RENT_APPROVAL_DATE"] = pd.to_datetime(test["RENT_APPROVAL_DATE"])
    return train, test


def temporal_split(train: pd.DataFrame, n_validation_months: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    months = sorted(train["RENT_APPROVAL_DATE"].dt.to_period("M").unique())
    if len(months) <= n_validation_months:
        raise ValueError("Not enough training months to build a temporal validation set.")
    validation_months = set(months[-n_validation_months:])
    mask = train["RENT_APPROVAL_DATE"].dt.to_period("M").isin(validation_months)
    return train.loc[~mask].copy(), train.loc[mask].copy()


def rmse(y_true: pd.Series, prediction: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, prediction)))


def global_median_predict(fit_df: pd.DataFrame, query_df: pd.DataFrame) -> np.ndarray:
    return np.full(len(query_df), fit_df[TARGET].median(), dtype=float)


def town_flat_median_predict(fit_df: pd.DataFrame, query_df: pd.DataFrame) -> np.ndarray:
    group_columns = ["TOWN", "FLAT_TYPE"]
    lookup = fit_df.groupby(group_columns)[TARGET].median()
    query_index = pd.MultiIndex.from_frame(query_df[group_columns])
    prediction = lookup.reindex(query_index).to_numpy(dtype=float)
    prediction = np.where(np.isnan(prediction), fit_df[TARGET].median(), prediction)
    return prediction


def build_ridge_pipeline(
    categorical_features: list[str] | None = None,
    numeric_features: list[str] | None = None,
) -> Pipeline:
    """Keep the original baseline defaults, with explicit schemas for ablations."""
    categorical_features = CATEGORICAL_FEATURES if categorical_features is None else categorical_features
    numeric_features = NUMERIC_FEATURES if numeric_features is None else numeric_features
    transformer = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=2,
                    sparse_output=True,
                ),
                categorical_features,
            ),
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("preprocess", transformer),
            # The one-hot design already contains category-level offsets.  A
            # zero-intercept sparse solve avoids unstable intercept recovery
            # when block and street columns are highly collinear.
            ("model", Ridge(alpha=10.0, solver="lsqr", tol=1e-4, fit_intercept=False)),
        ]
    )


def catboost_available() -> bool:
    try:
        import catboost  # noqa: F401
    except ImportError:
        return False
    return True


def train_catboost(
    fit_df: pd.DataFrame,
    query_df: pd.DataFrame,
    validation_df: pd.DataFrame | None = None,
) -> tuple[np.ndarray, dict]:
    from catboost import CatBoostRegressor

    model = CatBoostRegressor(
        iterations=800,
        depth=8,
        learning_rate=0.05,
        loss_function="RMSE",
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
    )
    fit_kwargs = {
        "X": fit_df[FEATURES],
        "y": fit_df[TARGET],
        "cat_features": CATEGORICAL_FEATURES,
    }
    if validation_df is not None:
        fit_kwargs["eval_set"] = (validation_df[FEATURES], validation_df[TARGET])
        fit_kwargs["use_best_model"] = True
        fit_kwargs["early_stopping_rounds"] = 60
    model.fit(**fit_kwargs)
    prediction = np.asarray(model.predict(query_df[FEATURES]), dtype=float)
    config = {
        "backend": "catboost",
        "iterations": int(model.tree_count_),
        "depth": 8,
        "learning_rate": 0.05,
        "early_stopping_rounds": 60 if validation_df is not None else None,
    }
    return prediction, config


def train_ridge(fit_df: pd.DataFrame, query_df: pd.DataFrame) -> tuple[np.ndarray, dict]:
    model = build_ridge_pipeline()
    model.fit(fit_df[FEATURES], fit_df[TARGET])
    prediction = np.asarray(model.predict(query_df[FEATURES]), dtype=float)
    return prediction, {
        "backend": "sklearn_ridge_fallback",
        "alpha": 10.0,
        "solver": "lsqr",
        "tol": 1e-4,
        "fit_intercept": False,
        "one_hot_min_frequency": 2,
    }


def train_supervised(
    fit_df: pd.DataFrame,
    query_df: pd.DataFrame,
    validation_df: pd.DataFrame | None,
    requested_model: str,
) -> tuple[np.ndarray, dict]:
    use_catboost = requested_model in {"auto", "catboost"} and catboost_available()
    if requested_model == "catboost" and not use_catboost:
        raise RuntimeError(
            "--model catboost was requested, but CatBoost is not installed. "
            "Install it with `python -m pip install catboost` or use --model ridge."
        )
    if use_catboost:
        return train_catboost(fit_df, query_df, validation_df)
    return train_ridge(fit_df, query_df)


def split_metadata(name: str, fit_df: pd.DataFrame, valid_df: pd.DataFrame) -> dict:
    return {
        "split": name,
        "fit_rows": len(fit_df),
        "validation_rows": len(valid_df),
        "fit_start": fit_df["RENT_APPROVAL_DATE"].min().strftime("%Y-%m"),
        "fit_end": fit_df["RENT_APPROVAL_DATE"].max().strftime("%Y-%m"),
        "validation_start": valid_df["RENT_APPROVAL_DATE"].min().strftime("%Y-%m"),
        "validation_end": valid_df["RENT_APPROVAL_DATE"].max().strftime("%Y-%m"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train HDB rent baseline models.")
    parser.add_argument(
        "--model",
        choices=["auto", "catboost", "ridge"],
        default="auto",
        help="auto uses CatBoost when installed, otherwise the Ridge fallback.",
    )
    parser.add_argument("--validation-months", type=int, default=3)
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    train, test = load_tables()
    temporal_fit, temporal_valid = temporal_split(train, args.validation_months)
    random_fit, random_valid = train_test_split(train, test_size=0.2, random_state=42)

    results: list[dict] = []
    baseline_methods = {
        "global_median": global_median_predict,
        "town_flat_median": town_flat_median_predict,
    }
    for split_name, fit_df, valid_df in [
        ("temporal", temporal_fit, temporal_valid),
        ("random", random_fit, random_valid),
    ]:
        metadata = split_metadata(split_name, fit_df, valid_df)
        for method_name, method in baseline_methods.items():
            prediction = method(fit_df, valid_df)
            results.append(
                {
                    **metadata,
                    "model": method_name,
                    "rmse": rmse(valid_df[TARGET], prediction),
                    "configuration": json.dumps({"fallback": method_name}, ensure_ascii=False),
                    "features": json.dumps([], ensure_ascii=False),
                }
            )

        prediction, config = train_supervised(
            fit_df, valid_df, temporal_valid if split_name == "temporal" and config_backend_is_catboost(args.model) else None, args.model
        )
        results.append(
            {
                **metadata,
                "model": config["backend"],
                "rmse": rmse(valid_df[TARGET], np.maximum(prediction, 0)),
                "configuration": json.dumps(config, ensure_ascii=False),
                "features": json.dumps(FEATURES, ensure_ascii=False),
            }
        )

    # Fit the selected supervised model on all labelled rows and generate the
    # competition submission in the same row order as test_clean.csv. The
    # supplied test table has no Id column, so its row position becomes the
    # required Id.
    final_prediction, final_config = train_supervised(train, test, None, args.model)
    final_prediction = np.maximum(final_prediction, 0)
    model_name = final_config["backend"]
    submission_path = SUBMISSION_DIR / f"submission_{model_name}.csv"
    pd.DataFrame(
        {
            "Id": np.arange(len(test), dtype=int),
            "Predicted": final_prediction,
        }
    ).to_csv(submission_path, index=False)

    results_df = pd.DataFrame(results)
    results_df.to_csv(REPORT_DIR / "baseline_results.csv", index=False)
    (REPORT_DIR / "baseline_config.json").write_text(
        json.dumps(
            {
                "selected_model": final_config,
                "features": FEATURES,
                "categorical_features": CATEGORICAL_FEATURES,
                "numeric_features": NUMERIC_FEATURES,
                "primary_validation": "latest approval months in labelled training data",
                "validation_months": args.validation_months,
                "random_split": {"test_size": 0.2, "random_state": 42},
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print("Baseline training complete.")
    print(results_df[["split", "model", "rmse"]].to_string(index=False))
    print(f"  Model used: {model_name}")
    print(f"  Submission: {submission_path}")


def config_backend_is_catboost(requested_model: str) -> bool:
    """Return whether validation may use CatBoost early stopping."""

    return requested_model == "catboost" or (requested_model == "auto" and catboost_available())


if __name__ == "__main__":
    main()
