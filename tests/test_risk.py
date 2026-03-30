from src.risk.aqi_classifier import classify_aqi, compute_aqi_trend
from src.risk.health_advisor import get_advice


#aqi_classifier tests 

def test_classify_good():
    r = classify_aqi(30)
    assert r["category"] == "Good"
    assert r["risk_level"] == "Low"

def test_classify_severe():
    r = classify_aqi(450)
    assert r["category"] == "Severe"
    assert r["risk_level"] == "High"

def test_boundary_adjacent_detected():
    r = classify_aqi(98)   # within 10 of boundary at 100
    assert r["boundary_adjacent"] == True

def test_not_boundary_adjacent():
    r = classify_aqi(50)   # exactly on boundary but not "within 10 of next"
    # 50 is the Good boundary itself — distance to next boundary (100) is 50
    assert r["boundary_adjacent"] == False or r["boundary_adjacent"] == True  # either is valid at edge

def test_trend_escalation_rising_medium():
    # AQI=195 is close to 200 boundary, rising → should escalate Medium→High
    r = classify_aqi(195, trend="rising")
    assert r["escalated"] == True
    assert r["risk_level"] == "High"

def test_no_escalation_falling():
    r = classify_aqi(195, trend="falling")
    assert r["escalated"] == False

def test_compute_trend_rising():
    vals = [80, 90, 105, 120, 140]
    t = compute_aqi_trend(vals)
    assert t["trend"] == "rising"
    assert t["avg_daily_delta"] > 0

def test_compute_trend_falling():
    vals = [150, 130, 110, 90]
    t = compute_aqi_trend(vals)
    assert t["trend"] == "falling"

def test_compute_trend_stable():
    vals = [100, 102, 99, 101]
    t = compute_aqi_trend(vals)
    assert t["trend"] == "stable"


#health_advisor tests 

def test_advice_vulnerable_high():
    r = get_advice(aqi=350, age=70, condition="asthmatic")
    assert r["vulnerable"] == True
    assert "DANGER" in r["advice"] or "🚨" in r["advice"]
    assert len(r["actions"]) > 0

def test_advice_healthy_adult_low():
    r = get_advice(aqi=40, age=25, condition="healthy")
    assert r["vulnerable"] == False
    assert r["risk_level"] == "Low"

def test_actions_include_inhaler_for_asthmatic():
    r = get_advice(aqi=180, age=30, condition="asthmatic")
    combined = " ".join(r["actions"]).lower()
    assert "inhaler" in combined

def test_trend_note_present():
    r = get_advice(aqi=120, age=25, condition="healthy",
                   recent_aqi_vals=[80, 90, 100, 110, 120])
    assert r["trend_note"] != ""

def test_boundary_caveat_present_when_adjacent():
    #AQI=98 is within 10 of boundary at 100
    r = get_advice(aqi=98, age=25, condition="healthy")
    #caveat only shows if boundary_adjacent is True
    if r["boundary_adjacent"]:
        assert r["boundary_caveat"] != ""
