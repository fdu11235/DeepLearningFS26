"""Feature engineering ported from the legacy notebook.

Adds: cyclical sin/cos for hour/weekday/month, is_weekend, age (years).
Drops: PII, raw temporal/location columns no longer needed.

`cc_num` is preserved here because the LSTM uses it to group transactions
into per-card sequences. It is dropped later (just before tensor construction).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# Reference date for age computation. The legacy notebook used pd.Timestamp("now"),
# which is non-deterministic across runs. We pin a fixed date here so age values
# are reproducible. This affects only the absolute scale of `age`; relative ordering
# and downstream model performance are unchanged.
AGE_REFERENCE_DATE = pd.Timestamp("2024-01-01")

# Columns to drop before modeling. cc_num/trans_date_trans_time are kept
# (they are removed later in the LSTM pipeline after sequence construction).
PII_AND_REDUNDANT = [
    "first",
    "last",
    "street",
    "trans_num",
    "unix_time",
    "lat",
    "long",
    "merch_lat",
    "merch_long",
    "dob",
    "hour",
    "weekday",
    "month",
]


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the legacy notebook's feature engineering, keeping `cc_num` and
    `trans_date_trans_time` so the LSTM can build temporal sequences."""
    df = df.copy()

    if not pd.api.types.is_datetime64_any_dtype(df["trans_date_trans_time"]):
        df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])

    df["hour"] = df["trans_date_trans_time"].dt.hour
    df["weekday"] = df["trans_date_trans_time"].dt.weekday
    df["month"] = df["trans_date_trans_time"].dt.month
    df["is_weekend"] = df["weekday"].isin([5, 6]).astype(int)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12)

    df["dob"] = pd.to_datetime(df["dob"])
    df["age"] = (AGE_REFERENCE_DATE - df["dob"]).dt.days // 365

    df = df.drop(columns=[c for c in PII_AND_REDUNDANT if c in df.columns])
    return df
