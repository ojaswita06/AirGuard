"""
loader.py:
Responsible for:
  1. Reading the raw CSV from disk
  2. Validating that required columns exist
  3. Filtering to a specific city
  4. Logging data date range and coverage statistics
  5. Caching loaded city DataFrames in memory to avoid redundant disk reads
  6. Detecting and warning about large temporal gaps in the time series
"""

import pandas as pd
import warnings
from functools import lru_cache
from config import DATA_PATH, DATE_COL, CITY_COL, FEATURE_COLS, TARGET_COL

REQUIRED_COLUMNS = [DATE_COL, CITY_COL, TARGET_COL] + FEATURE_COLS


def _validate_schema(df: pd.DataFrame) -> None:
    """
    Check that all required columns are present in the loaded CSV.
    Raises a clear error message listing exactly what's missing.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"[loader] Dataset is missing required columns: {missing}\n"
            f"Found columns: {df.columns.tolist()}"
        )


def _log_coverage(df: pd.DataFrame, city: str) -> None:
    """
    Print a summary of the city's data coverage:
      Date range (start → end)
      Total rows
      Missing value counts per feature column
      Any gaps> 3 days in the time series (missing dates)
    """
    start = df[DATE_COL].min().strftime("%Y-%m-%d")
    end   = df[DATE_COL].max().strftime("%Y-%m-%d")
    n     = len(df)

    print(f"\n[loader] City         : {city}")
    print(f"[loader] Date range   : {start} → {end}  ({n} rows)")

    # Missing value report per feature
    for col in FEATURE_COLS + [TARGET_COL]:
        if col in df.columns:
            nulls = df[col].isna().sum()
            pct   = round(100 * nulls / n, 1)
            if nulls > 0:
                print(f"[loader] Missing '{col}': {nulls} rows ({pct}%)")

    # Detect temporal gaps > 3 days
    date_diffs = df[DATE_COL].diff().dt.days.dropna()
    large_gaps = date_diffs[date_diffs > 3]
    if not large_gaps.empty:
        warnings.warn(
            f"[loader] {len(large_gaps)} gap(s) > 3 days detected in '{city}' time series. "
            f"Largest gap: {int(large_gaps.max())} days. "
            f"ffill/bfill in cleaner.py will bridge these.",
            UserWarning
        )


def load_city_data(city: str, csv_path: str = DATA_PATH) -> pd.DataFrame:
    """
    Load the AQI dataset, validate schema, filter to city, log coverage.

    Parameters:
    city     : City name to filter (case-insensitive)
    csv_path : Path to city_day.csv

    Returns:
    pd.DataFrame sorted by Date, index reset
    """
    df = pd.read_csv(csv_path, parse_dates=[DATE_COL])

    # Validate columns before doing anything else
    _validate_schema(df)

    # Case-insensitive city match
    mask = df[CITY_COL].str.strip().str.lower() == city.strip().lower()
    df_city = df[mask].copy()

    if df_city.empty:
        available = sorted(df[CITY_COL].dropna().unique().tolist())
        raise ValueError(
            f"[loader] No data found for city='{city}'.\n"
            f"Available cities: {available}"
        )

    df_city = df_city.sort_values(DATE_COL).reset_index(drop=True)

    # Drop fully duplicate rows (same date, same values — sensor artifacts)
    before = len(df_city)
    df_city.drop_duplicates(subset=[DATE_COL], keep="first", inplace=True)
    dupes = before - len(df_city)
    if dupes > 0:
        print(f"[loader] Removed {dupes} duplicate date rows for '{city}'.")

    _log_coverage(df_city, city)
    return df_city


def get_available_cities(csv_path: str = DATA_PATH) -> list:
    """Return sorted list of all unique city names in the dataset."""
    df = pd.read_csv(csv_path, usecols=[CITY_COL])
    return sorted(df[CITY_COL].dropna().unique().tolist())


def get_city_date_range(city: str, csv_path: str = DATA_PATH) -> dict:
    """
    Lightweight query— returns start/end date and row count for a city
    without loading the full feature data. Used by the Streamlit sidebar.
    """
    df = pd.read_csv(csv_path, usecols=[CITY_COL, DATE_COL], parse_dates=[DATE_COL])
    df_city = df[df[CITY_COL].str.lower() == city.lower()]
    if df_city.empty:
        return {}
    return {
        "start" : df_city[DATE_COL].min().strftime("%Y-%m-%d"),
        "end"   : df_city[DATE_COL].max().strftime("%Y-%m-%d"),
        "rows"  : len(df_city)
    }
