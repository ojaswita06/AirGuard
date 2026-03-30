"""
feature_engineer.py:
Creates additional input signals that help the LSTM learn temporal patterns:

  Rolling statistics:
    rolling_mean_N: average pollutant/AQI over past N days
    rolling_std_N : volatility of AQI over past N days
    These help the model understand trends and variance, not just point values.

  Lag features:
    AQI_lag_K: AQI value K days ago
    Direct autoregressive signal — yesterday's AQI is a strong predictor.

  Cyclical date encoding:
    sin/cos of day-of-year: encodes seasonality without discontinuities
      (e.g., day 365 and day 1 are close, which a raw integer can't express)

All NaNs introduced by shifting/rolling are dropped at the end.
"""

import numpy as np
import pandas as pd
from config import FEATURE_COLS, TARGET_COL, DATE_COL, ROLLING_WINDOWS, LAG_DAYS


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    #Rolling statistics on AQI 
    for w in ROLLING_WINDOWS:
        df[f"AQI_rolling_mean_{w}d"] = df[TARGET_COL].rolling(w).mean()
        df[f"AQI_rolling_std_{w}d"]  = df[TARGET_COL].rolling(w).std()

    #Lag features for AQI
    for lag in LAG_DAYS:
        df[f"AQI_lag_{lag}d"] = df[TARGET_COL].shift(lag)

    #Cyclical date encoding
    day_of_year = df[DATE_COL].dt.dayofyear
    df["day_sin"] = np.sin(2 * np.pi * day_of_year / 365)
    df["day_cos"] = np.cos(2 * np.pi * day_of_year / 365)

    #Drop rows with NaN introduced by rolling/shift operations
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def get_all_feature_cols(df: pd.DataFrame) -> list:
    """
    Return all columns to be used as model input features.
    Excludes Date, City, AQI_Bucket, and the raw TARGET_COL.
    """
    exclude = {DATE_COL, "City", "AQI_Bucket", TARGET_COL}
    return [c for c in df.columns if c not in exclude]
