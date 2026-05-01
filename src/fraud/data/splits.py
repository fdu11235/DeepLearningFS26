"""Time-based train/validation split.

The legacy notebook used a random stratified split, which leaks future
patterns into training. For the LSTM we use a chronological cutoff so the
validation set always lies after training in time — closer to deployment.
"""
from __future__ import annotations

import pandas as pd


def temporal_split(
    df: pd.DataFrame,
    val_fraction: float = 0.15,
    time_col: str = "trans_date_trans_time",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train_df, val_df) where val_df is the most recent `val_fraction`
    of transactions by `time_col`."""
    if time_col not in df.columns:
        raise KeyError(f"{time_col} not in dataframe columns")
    df_sorted = df.sort_values(time_col, kind="mergesort").reset_index(drop=True)
    cutoff_idx = int(len(df_sorted) * (1.0 - val_fraction))
    train_df = df_sorted.iloc[:cutoff_idx].copy()
    val_df = df_sorted.iloc[cutoff_idx:].copy()
    return train_df, val_df
