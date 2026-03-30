"""
config.py:
Single source of truth for all hyperparameters, file paths, and constants.
Changing a value here propagates everywhere — no magic numbers buried in code.
"""

import os

#Paths
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(BASE_DIR, "data", "city_day.csv")
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs")
MODEL_DIR   = os.path.join(OUTPUT_DIR, "models")
SCALER_DIR  = os.path.join(OUTPUT_DIR, "scalers")
PLOT_DIR    = os.path.join(OUTPUT_DIR, "plots")

MODEL_PATH  = os.path.join(MODEL_DIR,  "bilstm_aqi.keras")
SCALER_PATH = os.path.join(SCALER_DIR, "feature_scaler.pkl")
TARGET_SCALER_PATH = os.path.join(SCALER_DIR, "target_scaler.pkl")

#Data
FEATURE_COLS  = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]
TARGET_COL    = "AQI"
DATE_COL      = "Date"
CITY_COL      = "City"
WINDOW_SIZE   = 30          # days of history fed into LSTM
TEST_SPLIT    = 0.15        # 15% of data held out for evaluation
VAL_SPLIT     = 0.15        # 15% of training data used for validation

#Feature Engineering
ROLLING_WINDOWS   = [3, 7, 14]   # rolling mean/std windows (days)
LAG_DAYS          = [1, 2, 3, 7] # lag features for AQI

#Model
LSTM_UNITS_1   = 128
LSTM_UNITS_2   = 64
DENSE_UNITS    = 32
DROPOUT_RATE   = 0.3
LEARNING_RATE  = 0.001
BATCH_SIZE     = 32
EPOCHS         = 50
PATIENCE       = 8           # early stopping patience

#Risk Thresholds
AQI_GOOD        = 50
AQI_SATISFACTORY = 100
AQI_MODERATE    = 200
AQI_POOR        = 300
AQI_VERY_POOR   = 400
# > 400 = Severe
