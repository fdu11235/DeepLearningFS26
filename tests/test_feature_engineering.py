import numpy as np
import pandas as pd

from fraud.data.feature_engineering import feature_engineering


def _toy_df():
    return pd.DataFrame(
        {
            "trans_date_trans_time": pd.to_datetime(
                ["2020-01-01 03:00:00", "2020-06-13 22:30:00", "2020-12-31 23:59:00"]
            ),
            "cc_num": [111, 111, 222],
            "merchant": ["m1", "m2", "m1"],
            "category": ["food_dining", "shopping_pos", "food_dining"],
            "amt": [10.0, 99.0, 5.0],
            "first": ["A", "A", "B"],
            "last": ["X", "X", "Y"],
            "gender": ["M", "M", "F"],
            "street": ["s1", "s1", "s2"],
            "city": ["c1", "c1", "c2"],
            "state": ["NY", "NY", "CA"],
            "zip": [10001, 10001, 90001],
            "lat": [40.0, 40.0, 34.0],
            "long": [-74.0, -74.0, -118.0],
            "city_pop": [1000000, 1000000, 500000],
            "job": ["j1", "j1", "j2"],
            "dob": ["1990-01-15", "1990-01-15", "1985-06-01"],
            "trans_num": ["t1", "t2", "t3"],
            "unix_time": [1577847600, 1592087400, 1609459140],
            "merch_lat": [40.1, 40.2, 34.1],
            "merch_long": [-74.1, -74.2, -118.1],
            "is_fraud": [0, 1, 0],
        }
    )


def test_cyclical_features_added():
    df = feature_engineering(_toy_df())
    for c in ["hour_sin", "hour_cos", "weekday_sin", "weekday_cos", "month_sin", "month_cos"]:
        assert c in df.columns
    assert -1.0 <= df["hour_sin"].min() <= df["hour_sin"].max() <= 1.0


def test_is_weekend_flag():
    df = feature_engineering(_toy_df())
    # 2020-06-13 was a Saturday (weekday=5)
    assert df.loc[1, "is_weekend"] == 1
    # 2020-01-01 was a Wednesday
    assert df.loc[0, "is_weekend"] == 0


def test_age_is_positive():
    df = feature_engineering(_toy_df())
    assert (df["age"] > 0).all()


def test_pii_columns_dropped_but_grouping_keys_kept():
    df = feature_engineering(_toy_df())
    for dropped in ["first", "last", "street", "trans_num", "unix_time", "lat", "long", "merch_lat", "merch_long", "dob", "hour", "weekday", "month"]:
        assert dropped not in df.columns
    assert "cc_num" in df.columns
    assert "trans_date_trans_time" in df.columns
