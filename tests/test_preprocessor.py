import numpy as np
import pandas as pd

from fraud.data.feature_engineering import feature_engineering
from fraud.preprocessing import FEATURE_COLS, build_preprocessor


def _toy_df(n: int = 50, seed: int = 0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "trans_date_trans_time": pd.to_datetime("2020-01-01") + pd.to_timedelta(rng.integers(0, 365 * 24, n), unit="h"),
            "cc_num": rng.integers(100, 110, n),
            "merchant": rng.choice(["m1", "m2", "m3"], n),
            "category": rng.choice(["food_dining", "shopping_pos", "gas_transport"], n),
            "amt": rng.uniform(1, 500, n),
            "first": "A",
            "last": "B",
            "gender": rng.choice(["M", "F"], n),
            "street": "s",
            "city": rng.choice(["c1", "c2"], n),
            "state": rng.choice(["NY", "CA"], n),
            "zip": rng.choice([10001, 90001], n),
            "lat": 40.0,
            "long": -74.0,
            "city_pop": rng.integers(1000, 1_000_000, n),
            "job": rng.choice(["j1", "j2"], n),
            "dob": "1990-01-01",
            "trans_num": [f"t{i}" for i in range(n)],
            "unix_time": rng.integers(1577847600, 1609459200, n),
            "merch_lat": 40.1,
            "merch_long": -74.1,
            "is_fraud": rng.choice([0, 1], n, p=[0.8, 0.2]),
        }
    )


def test_preprocessor_fit_transform_no_nans():
    df = feature_engineering(_toy_df(n=100))
    pre = build_preprocessor()
    X = pre.fit_transform(df[FEATURE_COLS], df["is_fraud"])
    assert not np.isnan(np.asarray(X, dtype=float)).any()


def test_preprocessor_train_only_then_apply_to_val():
    """Fit on train slice, apply to validation slice. Output dim must match."""
    df = feature_engineering(_toy_df(n=200))
    train_df, val_df = df.iloc[:150], df.iloc[150:]
    pre = build_preprocessor()
    X_train = pre.fit_transform(train_df[FEATURE_COLS], train_df["is_fraud"])
    X_val = pre.transform(val_df[FEATURE_COLS])
    assert X_train.shape[1] == X_val.shape[1]
    assert X_val.shape[0] == 50
