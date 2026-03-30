"""
inference.py:
Loads saved model + scalers from disk and runs prediction on new input.
Also extracts recent AQI history so health_advisor can compute trend direction.

Used exclusively by app.py— completely decoupled from training code.
"""

import numpy as np
import joblib
import tensorflow as tf
import keras

from src.data.loader           import load_city_data
from src.data.cleaner          import clean_dataframe
from src.data.feature_engineer import engineer_features, get_all_feature_cols
from config import (
    MODEL_PATH, SCALER_PATH, TARGET_SCALER_PATH, WINDOW_SIZE, TARGET_COL
)


def load_artifacts():
    """Loading model and both scalers from disk."""
    model          = tf.keras.models.load_model(MODEL_PATH)
    feature_scaler = joblib.load(SCALER_PATH)
    target_scaler  = joblib.load(TARGET_SCALER_PATH)
    return model, feature_scaler, target_scaler


def predict_next_day(city: str, model, feature_scaler, target_scaler) -> dict:
    """
    Full inference pipeline for a given city.
    Returns:
    dict with:
      predicted_aqi: float— next-day AQI prediction
      recent_aqi_vals: list — last 7 days of actual AQI (for trend)
    """
    df = load_city_data(city)
    df = clean_dataframe(df)
    df = engineer_features(df)
    feature_cols = get_all_feature_cols(df)

    if len(df) < WINDOW_SIZE:
        raise ValueError(
            f"Not enough data for city='{city}'. Need≥ {WINDOW_SIZE} rows after cleaning."
        )

    # Extract last 7 raw AQI values before scaling (for trend computation)
    recent_aqi_vals = df[TARGET_COL].values[-7:].tolist()

    # Scale — transform only (scaler was fit during training, never refit here)
    df[feature_cols] = feature_scaler.transform(df[feature_cols])
    df[[TARGET_COL]]  = target_scaler.transform(df[[TARGET_COL]])

    # Build inference window
    window = df[feature_cols].values[-WINDOW_SIZE:]
    X = window.reshape(1, WINDOW_SIZE, len(feature_cols))

    y_norm = model.predict(X, verbose=0).flatten()[0]
    y_aqi  = target_scaler.inverse_transform([[y_norm]])[0][0]

    return {
        "predicted_aqi"  : round(float(y_aqi), 2),
        "recent_aqi_vals": recent_aqi_vals,
    }
