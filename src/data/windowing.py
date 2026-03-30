"""
windowing.py:
Converts the flat time-series DataFrame into 3D arrays that LSTMs expect.

Sliding window logic:
  For each position i starting at WINDOW_SIZE:
    X[i] = feature rows from (i - WINDOW_SIZE) to (i - 1)  → shape (30, n_features)
    y[i] = AQI value at row i                               → the next-day label

So if WINDOW_SIZE = 30:
  X[0] is days 0–29, y[0] is day 30's AQI
  X[1] is days 1–30, y[1] is day 31's AQI
  ... and so on.

Final shapes:
  X -> (n_samples, WINDOW_SIZE, n_features)
  y -> (n_samples,)
"""

import numpy as np
import pandas as pd
from config import TARGET_COL, WINDOW_SIZE


def build_windows(
    df: pd.DataFrame,
    feature_cols: list,
    window_size: int = WINDOW_SIZE
):
    """
    Build sliding-window (X, y) pairs from a feature-engineered DataFrame.

    Parameters:
    df           : Cleaned, feature-engineered DataFrame
    feature_cols : List of column names to use as input features
    window_size  : Number of past days to include in each window

    Returns:
    X: np.ndarray of shape (n_samples, window_size, n_features)
    y: np.ndarray of shape (n_samples,)
    """
    feature_array = df[feature_cols].values   # shape: (n_rows, n_features)
    target_array  = df[TARGET_COL].values     # shape: (n_rows,)

    X, y = [], []

    for i in range(window_size, len(feature_array)):
        X.append(feature_array[i - window_size : i])  # 30 rows of features
        y.append(target_array[i])                      # next day's AQI

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def train_test_split_temporal(X, y, test_split: float):
    """
    Time-aware split— never shuffle! Future must not leak into training.
    Takes the last `test_split` fraction as the test set.
    """
    split_idx = int(len(X) * (1 - test_split))
    return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]
