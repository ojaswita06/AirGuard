"""
cleaner.py:
Handles data quality issues:
  1. Drops rows where AQI itself is missing (no label → can't train)
  2. For pollutant features: fills nulls using forward-fill then backward-fill
     (uses the nearest real measurement in time — better than mean for time-series)
  3. Clips extreme outliers at the 1st and 99th percentile per column
     (sensor spikes can otherwise dominate the loss function)
"""

import numpy as np
import pandas as pd
from config import FEATURE_COLS, TARGET_COL


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full cleaning pipeline. Returns a cleaned copy.
    Steps:
    1. Drop rows with missing AQI (target)
    2. Forward-fill then backward-fill pollutant features
    3. Clip outliers at [1%, 99%] per feature column
    """
    df = df.copy()

    # Step 1: AQI is the label — rows without it are useless
    before = len(df)
    df.dropna(subset=[TARGET_COL], inplace=True)
    dropped = before - len(df)
    if dropped > 0:
        print(f"[cleaner] Dropped {dropped} rows with missing AQI.")

    # Step 2: Fill pollutant nulls via time-aware forward/backward fill
    for col in FEATURE_COLS:
        if col in df.columns:
            null_count = df[col].isna().sum()
            if null_count > 0:
                df[col] = df[col].ffill().bfill()
                print(f"[cleaner] Filled {null_count} nulls in '{col}' via ffill/bfill.")

    # Step 3: Clip outliers — protects BiLSTM from extreme sensor noise
    for col in FEATURE_COLS + [TARGET_COL]:
        if col in df.columns:
            low  = df[col].quantile(0.01)
            high = df[col].quantile(0.99)
            df[col] = df[col].clip(lower=low, upper=high)

    df.reset_index(drop=True, inplace=True)
    return df
