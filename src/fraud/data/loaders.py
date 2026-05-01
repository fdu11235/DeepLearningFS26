"""CSV loaders with explicit dtypes and datetime parsing."""
from __future__ import annotations

from pathlib import Path
import pandas as pd


def load_raw_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    return df
