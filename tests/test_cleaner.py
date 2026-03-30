import pandas as pd
import numpy as np
from src.data.cleaner import clean_dataframe

def test_drops_missing_aqi():
    df = pd.DataFrame({"PM2.5": [10, 20], "PM10": [30, 40],
                        "NO2": [5,5], "SO2": [1,1], "CO": [0.5,0.5],
                        "O3": [20,20], "AQI": [None, 80]})
    result = clean_dataframe(df)
    assert len(result) == 1
    assert result["AQI"].iloc[0] == 80

def test_ffill_pollutants():
    df = pd.DataFrame({"PM2.5": [10, None, 30], "PM10": [20, None, 40],
                        "NO2": [5,5,5], "SO2": [1,1,1], "CO": [0.5,0.5,0.5],
                        "O3": [20,20,20], "AQI": [50, 60, 70]})
    result = clean_dataframe(df)
    assert result["PM2.5"].isna().sum() == 0
