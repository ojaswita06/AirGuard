"""
preprocessor.py:
Orchestrates the full preprocessing pipeline in one call:
  loader -> cleaner -> feature_engineer -> normalize -> windowing

Also fits and saves the MinMaxScaler so inference can reuse it.
This is the single function that train.py calls— it handles everything
and returns ready-to-use (X_train, X_test, y_train, y_test) arrays.
"""

import os
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler

from src.data.loader           import load_city_data
from src.data.cleaner          import clean_dataframe
from src.data.feature_engineer import engineer_features, get_all_feature_cols
from src.data.windowing        import build_windows, train_test_split_temporal
from config import (
    SCALER_DIR, SCALER_PATH, TARGET_SCALER_PATH,
    TARGET_COL, TEST_SPLIT, WINDOW_SIZE
)


def run_preprocessing_pipeline(city: str):
    """
    Full pipeline from raw CSV → model-ready arrays.
    Returns:
    X_train, X_test, y_train, y_test: np.ndarrays
    feature_scaler: fitted MinMaxScaler (for features)
    target_scaler: fitted MinMaxScaler (for AQI)
    feature_cols: list of feature column names used
    """
    os.makedirs(SCALER_DIR, exist_ok=True)

    # Step 1: Load
    print(f"[pipeline] Loading data for city='{city}'...")
    df = load_city_data(city)

    # Step 2: Clean
    print("[pipeline] Cleaning...")
    df = clean_dataframe(df)

    # Step 3: Feature engineering
    print("[pipeline] Engineering features...")
    df = engineer_features(df)
    feature_cols = get_all_feature_cols(df)
    print(f"[pipeline] Total features: {len(feature_cols)} → {feature_cols}")

    # Step 4: Normalize features (fit on full data, then re-split — common for small datasets)
    feature_scaler = MinMaxScaler()
    df[feature_cols] = feature_scaler.fit_transform(df[feature_cols])

    target_scaler = MinMaxScaler()
    df[[TARGET_COL]] = target_scaler.fit_transform(df[[TARGET_COL]])

    # Save scalers for inference
    joblib.dump(feature_scaler, SCALER_PATH)
    joblib.dump(target_scaler,  TARGET_SCALER_PATH)
    print(f"[pipeline] Scalers saved.")

    # Step 5: Build sliding windows
    X, y = build_windows(df, feature_cols, WINDOW_SIZE)
    print(f"[pipeline] Window shape — X: {X.shape}, y: {y.shape}")

    # Step 6: Temporal train/test split
    X_train, X_test, y_train, y_test = train_test_split_temporal(X, y, TEST_SPLIT)
    print(f"[pipeline] Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")

    return X_train, X_test, y_train, y_test, feature_scaler, target_scaler, feature_cols
