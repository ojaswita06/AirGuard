"""
health_advisor.py:
Generates a personalized, context-aware health advisory based on:

  1. Predicted AQI + its CPCB classification
  2. AQI trend direction (rising/ falling/ stable)
  3. User's age -> age group (child/ adult/ senior)
  4. User's health condition (healthy/ asthmatic/ diabetic/ elderly)
  5. Whether the prediction sits near a category boundary (uncertainty signal)

Design decisions:
  1. Vulnerability is additive: a child who is also asthmatic is flagged vulnerable
    via both age_group AND condition checks.
  2. Trend escalation: if AQI is rising and the user is vulnerable, the advice
    explicitly mentions worsening conditions expected tomorrow.
  3. Boundary caveat: if the model's prediction is boundary adjacent, the advice
    appends a note that the actual category could be one step higher or lower.
  4. Precautionary actions: each risk tier includes concrete, actionable steps
    (mask type, window guidance, inhaler reminder) rather than vague warnings.
"""

from src.risk.aqi_classifier import classify_aqi, compute_aqi_trend


CHILD_AGE  = 12
SENIOR_AGE = 60


def _get_age_group(age: int) -> str:
    if age < CHILD_AGE:
        return "child"
    elif age > SENIOR_AGE:
        return "senior"
    return "adult"


def _is_vulnerable(condition: str, age_group: str) -> bool:
    return condition in ("asthmatic", "diabetic", "elderly") or age_group in ("child", "senior")


def _precautionary_actions(risk_level: str, condition: str) -> list:
    """
    Returns a list of specific precautionary actions for a given risk level
    and health condition. These are displayed as bullet points in the app.
    """
    base_actions = {
        "Low": [
            "Normal outdoor activity is safe.",
            "Stay hydrated — air pollutants increase oxidative stress.",
        ],
        "Medium": [
            "Reduce prolonged outdoor exertion (jogging, cycling).",
            "Keep windows partially closed during peak traffic hours (8–10 AM, 6–9 PM).",
            "Consider a basic surgical mask if outside for > 2 hours.",
        ],
        "High": [
            "Avoid all non-essential outdoor exposure.",
            "Use an N95/N99 respirator if going out is unavoidable.",
            "Run an air purifier indoors (HEPA filter preferred).",
            "Keep all windows and doors sealed.",
        ]
    }

    condition_extras = {
        "asthmatic": {
            "Medium": ["Keep your rescue inhaler accessible at all times."],
            "High"  : ["Use your preventer inhaler as prescribed. Contact your doctor if symptoms worsen."],
        },
        "diabetic": {
            "Medium": ["Air pollution can affect blood sugar regulation — monitor more frequently."],
            "High"  : ["Pollution-induced inflammation may spike blood sugar. Check levels every 4–6 hours."],
        },
        "elderly": {
            "Medium": ["Avoid morning walks today — AQI is highest in early hours."],
            "High"  : ["Remain indoors entirely. Ask someone to run errands on your behalf."],
        }
    }

    actions = list(base_actions.get(risk_level, []))
    extras  = condition_extras.get(condition, {}).get(risk_level, [])
    return actions + extras


def _trend_note(trend: str, avg_delta: float, vulnerable: bool) -> str:
    """Build a short sentence about AQI trajectory to append to the advice."""
    if trend == "rising":
        note = f"⬆️ AQI is trending upward (~{abs(avg_delta):.1f} pts/day)."
        if vulnerable:
            note += " Conditions may worsen tomorrow — plan accordingly."
    elif trend == "falling":
        note = f"⬇️ AQI is improving (~{abs(avg_delta):.1f} pts/day). Relief likely tomorrow."
    else:
        note = "➡️ AQI is relatively stable over the past few days."
    return note


_CORE_ADVICE = {
    ("Low",    False): "Air quality is good. Outdoor activities are safe — enjoy the day!",
    ("Low",    True ): "Air quality is acceptable. Sensitive individuals may notice mild irritation during intense outdoor exertion.",
    ("Medium", False): "Air quality is moderate. Healthy adults should reduce prolonged outdoor exertion.",
    ("Medium", True ): "Moderate AQI is a meaningful concern for your health profile. Limit outdoor time and monitor symptoms.",
    ("High",   False): "Poor air quality. Avoid outdoor exercise. Wear an N95 mask if going out.",
    ("High",   True ): "DANGER: AQI is in a hazardous range for your health profile. Stay indoors and seek medical guidance if you experience breathing difficulty.",
}


def get_advice(
    aqi             : float,
    age             : int,
    condition       : str,
    recent_aqi_vals : list = None
) -> dict:
    """
    Generate a full structured health advisory.

    Parameters:
    aqi: Predicted next-day AQI
    age: User's age in years
    condition: 'healthy'| 'asthmatic'| 'diabetic'| 'elderly'
    recent_aqi_vals: Optional list of recent AQI values for trend computation

    Returns:
    dict with keys:
      aqi, category, risk_level, color, age_group, condition,
      vulnerable, escalated, boundary_adjacent, trend, avg_daily_delta,
      advice, actions, trend_note, boundary_caveat
    """
    # Computing trend from recent history
    trend_data = {"trend": "stable", "avg_daily_delta": 0.0}
    if recent_aqi_vals and len(recent_aqi_vals) >= 2:
        trend_data = compute_aqi_trend(recent_aqi_vals)

    # Classifying AQI with trend-awareness
    classification = classify_aqi(aqi, trend=trend_data["trend"])

    risk_level = classification["risk_level"]
    age_group  = _get_age_group(age)
    vulnerable = _is_vulnerable(condition, age_group)

    # Core advice sentence
    advice = _CORE_ADVICE.get((risk_level, vulnerable),
                               "Monitor local AQI updates and consult your doctor.")

    # Precautionary action list
    actions = _precautionary_actions(risk_level, condition)

    # Trend note
    trend_note = _trend_note(
        trend_data["trend"],
        trend_data["avg_daily_delta"],
        vulnerable
    )

    # Boundary caveat— model uncertainty signal
    boundary_caveat = (
        "Note: The predicted AQI sits near a category boundary. "
        "The actual risk level could be one tier higher or lower."
        if classification["boundary_adjacent"] else ""
    )

    return {
        "aqi"              : aqi,
        "category"         : classification["category"],
        "risk_level"       : risk_level,
        "color"            : classification["color"],
        "age_group"        : age_group,
        "condition"        : condition,
        "vulnerable"       : vulnerable,
        "escalated"        : classification["escalated"],
        "boundary_adjacent": classification["boundary_adjacent"],
        "trend"            : trend_data["trend"],
        "avg_daily_delta"  : trend_data["avg_daily_delta"],
        "advice"           : advice,
        "actions"          : actions,
        "trend_note"       : trend_note,
        "boundary_caveat"  : boundary_caveat,
    }
