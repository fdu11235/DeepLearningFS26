"""sklearn ColumnTransformer for the fraud features.

Mirrors the legacy notebook's column groups exactly:
  * Numerical (StandardScaler): amt, city_pop, age, hour/weekday/month sin-cos
  * Low-cardinality (OneHotEncoder, drop="first"): gender, state, category, is_weekend
  * High-cardinality (TargetEncoder, smooth="auto"): merchant, job, city, zip
"""
from __future__ import annotations

import pickle
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

NUM_COLS = [
    "amt",
    "city_pop",
    "age",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "month_sin",
    "month_cos",
]
LOW_CARD = ["gender", "state", "category", "is_weekend"]
HIGH_CARD = ["merchant", "job", "city", "zip"]
FEATURE_COLS = NUM_COLS + LOW_CARD + HIGH_CARD


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUM_COLS),
            (
                "low",
                OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
                LOW_CARD,
            ),
            (
                "high",
                TargetEncoder(smooth="auto", target_type="binary"),
                HIGH_CARD,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def save_preprocessor(preprocessor: ColumnTransformer, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(preprocessor, f)


def load_preprocessor(path: str | Path) -> ColumnTransformer:
    with open(path, "rb") as f:
        return pickle.load(f)
