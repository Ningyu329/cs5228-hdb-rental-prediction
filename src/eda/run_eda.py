"""Generate first-stage EDA tables and figures.

Run from the project root after cleaning:

    python src/data/clean_data.py
    python src/eda/run_eda.py

The analysis treats the observations as rental records with a time dimension,
not as a pure time-series signal.  HDB regions are joined only for analysis;
the original interim main table is not modified here.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"
TABLE_DIR = PROJECT_ROOT / "reports" / "tables"
MPL_CONFIG_DIR = PROJECT_ROOT / ".matplotlib"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt


def address_key(df: pd.DataFrame) -> pd.Series:
    return (
        df["TOWN"].astype("string").str.strip().str.lower()
        + "|"
        + df["BLOCK"].astype("string").str.strip().str.lower()
        + "|"
        + df["STREET"].astype("string").str.strip().str.lower()
    )


def save_figure(name: str) -> None:
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / name, dpi=160, bbox_inches="tight")
    plt.close()


def save_summary(df: pd.DataFrame, name: str) -> None:
    df.to_csv(TABLE_DIR / name, index=False)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(INTERIM_DIR / "train_clean.csv")
    hdb = pd.read_csv(INTERIM_DIR / "auxiliary" / "sg-hdb-block_clean.csv")

    train["RENT_APPROVAL_DATE"] = pd.to_datetime(train["RENT_APPROVAL_DATE"])
    numeric = [
        "FLOOR_AREA_SQM",
        "LEASE_COMMENCE_DATE",
        "RENT_YEAR",
        "RENT_MONTH",
        "RENT_MONTH_INDEX",
        "FLAT_AGE",
        "MONTHLY_RENT",
    ]
    for column in numeric:
        if column in train.columns:
            train[column] = pd.to_numeric(train[column], errors="coerce")

    train["ADDRESS_KEY"] = address_key(train)
    location = hdb[[
        "ADDRESS_KEY",
        "SUBZONE",
        "PLANNING_AREA",
        "REGION",
        "YEAR_COMPLETED",
    ]].drop_duplicates("ADDRESS_KEY")
    train = train.merge(location, on="ADDRESS_KEY", how="left", validate="many_to_one")
    train["REGION"] = train["REGION"].fillna("unknown")
    train["PLANNING_AREA"] = train["PLANNING_AREA"].fillna("unknown")
    return train, hdb


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    train, _ = load_data()

    # 1. Regional distribution and regional rent differences.
    region_summary = (
        train.groupby("REGION", dropna=False)["MONTHLY_RENT"]
        .agg(count="size", mean="mean", median="median", std="std")
        .sort_values("count", ascending=False)
        .reset_index()
    )
    save_summary(region_summary, "region_rent_summary.csv")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    region_order = region_summary["REGION"].tolist()
    axes[0].bar(region_order, region_summary["count"], color="#4c78a8")
    axes[0].set_title("Rental records by region")
    axes[0].set_ylabel("Number of records")
    axes[0].tick_params(axis="x", rotation=35)
    axes[1].bar(region_order, region_summary["median"], color="#f58518")
    axes[1].set_title("Median rent by region")
    axes[1].set_ylabel("Median monthly rent")
    axes[1].tick_params(axis="x", rotation=35)
    save_figure("01_region_distribution_and_rent.png")

    # 2. Monthly trend.
    monthly = (
        train.groupby("RENT_APPROVAL_DATE")["MONTHLY_RENT"]
        .agg(count="size", mean="mean", median="median", std="std")
        .reset_index()
    )
    save_summary(monthly, "monthly_rent_summary.csv")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(monthly["RENT_APPROVAL_DATE"], monthly["median"], marker="o", label="Median")
    ax.plot(monthly["RENT_APPROVAL_DATE"], monthly["mean"], marker="o", label="Mean")
    ax.set_title("Monthly rent trend")
    ax.set_xlabel("Rent approval month")
    ax.set_ylabel("Monthly rent")
    ax.legend()
    plt.xticks(rotation=45)
    save_figure("02_monthly_rent_trend.png")

    # 3. Town differences.
    town_summary = (
        train.groupby("TOWN")["MONTHLY_RENT"]
        .agg(count="size", mean="mean", median="median", std="std")
        .sort_values("median", ascending=True)
        .reset_index()
    )
    save_summary(town_summary, "town_rent_summary.csv")
    fig, ax = plt.subplots(figsize=(11, 9))
    ax.barh(town_summary["TOWN"], town_summary["median"], color="#59a14f")
    ax.set_title("Median monthly rent by town")
    ax.set_xlabel("Median monthly rent")
    save_figure("03_town_rent_difference.png")

    # 4. Flat type and flat model differences.
    flat_type_summary = (
        train.groupby("FLAT_TYPE")["MONTHLY_RENT"]
        .agg(count="size", mean="mean", median="median", std="std")
        .sort_values("median", ascending=True)
        .reset_index()
    )
    flat_model_summary = (
        train.groupby("FLAT_MODEL")["MONTHLY_RENT"]
        .agg(count="size", mean="mean", median="median", std="std")
        .sort_values("median", ascending=True)
        .reset_index()
    )
    save_summary(flat_type_summary, "flat_type_rent_summary.csv")
    save_summary(flat_model_summary, "flat_model_rent_summary.csv")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].barh(flat_type_summary["FLAT_TYPE"], flat_type_summary["median"], color="#e45756")
    axes[0].set_title("Median rent by flat type")
    axes[0].set_xlabel("Median monthly rent")
    axes[1].barh(flat_model_summary["FLAT_MODEL"], flat_model_summary["median"], color="#b279a2")
    axes[1].set_title("Median rent by flat model")
    axes[1].set_xlabel("Median monthly rent")
    save_figure("04_flat_type_and_model_difference.png")

    # 5. Area and age relationships.  The scatter uses a fixed sample only to
    # keep the plot readable; all summary tables use the full training set.
    area_bins = [0, 50, 70, 90, 110, 130, 160, np.inf]
    age_bins = [0, 10, 20, 30, 40, 50, 60, np.inf]
    train["AREA_BIN"] = pd.cut(train["FLOOR_AREA_SQM"], bins=area_bins, right=False)
    train["AGE_BIN"] = pd.cut(train["FLAT_AGE"], bins=age_bins, right=False)
    area_summary = (
        train.groupby("AREA_BIN", observed=False)["MONTHLY_RENT"]
        .agg(count="size", mean="mean", median="median")
        .reset_index()
    )
    age_summary = (
        train.groupby("AGE_BIN", observed=False)["MONTHLY_RENT"]
        .agg(count="size", mean="mean", median="median")
        .reset_index()
    )
    save_summary(area_summary, "floor_area_rent_summary.csv")
    save_summary(age_summary, "flat_age_rent_summary.csv")

    sample = train.sample(min(25_000, len(train)), random_state=42)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].scatter(sample["FLOOR_AREA_SQM"], sample["MONTHLY_RENT"], alpha=0.2, s=10)
    axes[0, 0].set_title("Floor area vs monthly rent")
    axes[0, 0].set_xlabel("Floor area (sqm)")
    axes[0, 0].set_ylabel("Monthly rent")
    axes[0, 1].plot(area_summary["AREA_BIN"].astype(str), area_summary["median"], marker="o")
    axes[0, 1].set_title("Median rent by floor-area bin")
    axes[0, 1].tick_params(axis="x", rotation=45)
    axes[1, 0].scatter(sample["FLAT_AGE"], sample["MONTHLY_RENT"], alpha=0.2, s=10)
    axes[1, 0].set_title("Flat age vs monthly rent")
    axes[1, 0].set_xlabel("Flat age (years)")
    axes[1, 0].set_ylabel("Monthly rent")
    axes[1, 1].plot(age_summary["AGE_BIN"].astype(str), age_summary["median"], marker="o")
    axes[1, 1].set_title("Median rent by flat-age bin")
    axes[1, 1].tick_params(axis="x", rotation=45)
    save_figure("05_floor_area_and_flat_age.png")

    train[["FLOOR_AREA_SQM", "FLAT_AGE", "MONTHLY_RENT"]].corr().to_csv(
        TABLE_DIR / "area_age_rent_correlations.csv"
    )

    print("EDA complete.")
    print(f"  Figures: {FIGURE_DIR}")
    print(f"  Tables:  {TABLE_DIR}")


if __name__ == "__main__":
    main()
