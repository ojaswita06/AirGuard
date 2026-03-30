import numpy as np
from src.data.windowing import build_windows, train_test_split_temporal
import pandas as pd

def _make_dummy_df(n=100):
    df = pd.DataFrame(np.random.rand(n, 7),
                      columns=["PM2.5","PM10","NO2","SO2","CO","O3","AQI"])
    return df

def test_window_shape():
    df = _make_dummy_df(100)
    X, y = build_windows(df, ["PM2.5","PM10","NO2","SO2","CO","O3"], window_size=30)
    assert X.shape == (70, 30, 6)
    assert y.shape == (70,)

def test_temporal_split_no_shuffle():
    X = np.arange(100).reshape(100, 1, 1).astype(float)
    y = np.arange(100).astype(float)
    Xtr, Xte, ytr, yte = train_test_split_temporal(X, y, test_split=0.2)
    assert ytr[-1] < yte[0]   # training ends before test begins
