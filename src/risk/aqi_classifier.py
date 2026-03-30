"""
aqi_classifier.py:
Maps a raw AQI value to India's official CPCB AQI categories.
Thresholds (CPCB India):
  0–50: Good
  51–100: Satisfactory
  101–200: Moderate
  201–300: Poor
  301–400: Very Poor
  401+ : Severe

Beyond basic classification this module also provides:

  Boundary proximity detection:
    If the predicted AQI is within ±10 units of a category boundary,
    the forecast is flagged as "boundary-adjacent" — meaning a small
    model error could shift the true category. The app uses this to
    show a confidence caveat to the user.

  Trend-aware severity escalation:
    Accepts an optional trend signal ('rising'| 'falling'| 'stable').
    If AQI is rising and sits near a boundary, the returned risk_level
    is escalated one tier (e.g., Medium -> High) as a precautionary signal.
    This mimics what a public health advisory system would do.

  AQI delta summary:
    Given a list of recent AQI values, computes the 3-day average delta
    so the app can display "AQI trending up/down by X points/day".
"""

import numpy as np
from config import (
    AQI_GOOD, AQI_SATISFACTORY, AQI_MODERATE, AQI_POOR, AQI_VERY_POOR
)

# All CPCB boundary points in ascending order
_BOUNDARIES = [AQI_GOOD, AQI_SATISFACTORY, AQI_MODERATE, AQI_POOR, AQI_VERY_POOR]
_BOUNDARY_PROXIMITY = 10   # units within which we flag a boundary-adjacent prediction


def classify_aqi(aqi: float, trend: str = "stable") -> dict:
    """
    Classify AQI into CPCB category with boundary proximity and trend escalation.

    Parameters:
    aqi: Predicted AQI value (float)
    trend: 'rising', 'falling', or 'stable' — derived from recent history

    Returns:
    dict with keys:
      category: CPCB category string
      risk_level: 'Low' | 'Medium' | 'High'
      color: Hex color for UI rendering
      boundary_adjacent: bool — True if AQI is within ±10 of a boundary
      escalated: bool — True if risk_level was raised due to rising trend
    """
    base = _base_classify(aqi)

    # Checking if AQI is within ±BOUNDARY_PROXIMITY of any category boundary
    boundary_adjacent = any(
        abs(aqi - b) <= _BOUNDARY_PROXIMITY for b in _BOUNDARIES
    )

    # Escalating risk if AQI is rising and near an upper boundary
    escalated = False
    if trend == "rising" and boundary_adjacent:
        if base["risk_level"] == "Low":
            base["risk_level"] = "Medium"
            escalated = True
        elif base["risk_level"] == "Medium":
            base["risk_level"] = "High"
            escalated = True

    return {
        **base,
        "boundary_adjacent": boundary_adjacent,
        "escalated"        : escalated,
        "trend"            : trend,
    }


def _base_classify(aqi: float) -> dict:
    """Core CPCB threshold lookup. Returns category, risk_level, color."""
    if aqi <= AQI_GOOD:
        return {"category": "Good",         "risk_level": "Low",    "color": "#4CAF50"}
    elif aqi <= AQI_SATISFACTORY:
        return {"category": "Satisfactory", "risk_level": "Low",    "color": "#8BC34A"}
    elif aqi <= AQI_MODERATE:
        return {"category": "Moderate",     "risk_level": "Medium", "color": "#FFC107"}
    elif aqi <= AQI_POOR:
        return {"category": "Poor",         "risk_level": "High",   "color": "#FF5722"}
    elif aqi <= AQI_VERY_POOR:
        return {"category": "Very Poor",    "risk_level": "High",   "color": "#F44336"}
    else:
        return {"category": "Severe",       "risk_level": "High",   "color": "#9C27B0"}


def compute_aqi_trend(recent_aqi_values: list) -> dict:
    """
    Given a list of recent AQI values (chronological order), compute:
      trend direction: 'rising' | 'falling' | 'stable'
      avg_daily_delta: mean day-over-day change (signed)

    Parameters:
    recent_aqi_values : list of floats, at least 2 values, most recent last

    Returns:
    dict with 'trend' and 'avg_daily_delta'
    """
    if len(recent_aqi_values) < 2:
        return {"trend": "stable", "avg_daily_delta": 0.0}

    deltas = np.diff(recent_aqi_values[-4:])   # use last 3 day-over-day changes
    avg_delta = float(np.mean(deltas))

    if avg_delta > 5:
        trend = "rising"
    elif avg_delta < -5:
        trend = "falling"
    else:
        trend = "stable"

    return {
        "trend"          : trend,
        "avg_daily_delta": round(avg_delta, 2)
    }
