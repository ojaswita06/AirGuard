"""
app.py:
AirGuard— Upgraded Streamlit Frontend
Features:
  AQI Gauge (SVG speedometer)
  7-day historical trend chart
  Pollutant breakdown bar chart
  Color-coded health advisory cards
  Demographic risk cards
  Model confidence range
  Professional dark theme layout
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime

from src.model.architecture    import build_bilstm_model
from src.model.trainer         import train_model
from src.model.evaluator       import evaluate_model
from src.data.loader           import get_available_cities, get_city_date_range, load_city_data
from src.data.cleaner          import clean_dataframe
from src.data.feature_engineer import engineer_features, get_all_feature_cols
from src.pipeline.inference    import load_artifacts, predict_next_day
from src.risk.health_advisor   import get_advice
from config                    import TARGET_COL

#Page config 
st.set_page_config(
    page_title="AirGuard — AQI Health Forecaster",
    layout="wide"
)

#Global CSS 
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=Space+Mono:wght@400;700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.block-container { padding-top: 2rem; max-width: 1200px; }

.metric-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid #0f3460;
    border-radius: 16px;
    padding: 20px;
    text-align: center;
    margin-bottom: 12px;
}
.metric-label {
    font-size: 0.75rem;
    color: #8892b0;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 6px;
    font-family: 'Space Mono', monospace;
}
.metric-value {
    font-size: 2rem;
    font-weight: 600;
    color: #e6f1ff;
    line-height: 1.1;
}
.metric-sub {
    font-size: 0.8rem;
    color: #64ffda;
    margin-top: 4px;
}

.advisory-card {
    border-radius: 16px;
    padding: 24px;
    margin: 12px 0;
    border-left: 6px solid;
}

.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #64ffda;
    margin-bottom: 16px;
    margin-top: 8px;
}

.profile-chip {
    display: inline-block;
    background: #0f3460;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.8rem;
    color: #ccd6f6;
    margin: 3px;
}

.source-note {
    font-size: 0.72rem;
    color: #4a5568;
    text-align: center;
    margin-top: 40px;
    font-family: 'Space Mono', monospace;
}
</style>
""", unsafe_allow_html=True)


#AQI helpers 
AQI_BANDS = [
    (0,   50,  "#00e400", "Good"),
    (51,  100, "#ffff00", "Satisfactory"),
    (101, 200, "#ff7e00", "Moderate"),
    (201, 300, "#ff0000", "Poor"),
    (301, 400, "#8f3f97", "Very Poor"),
    (401, 500, "#7e0023", "Severe"),
]

def aqi_color_category(aqi):
    for lo, hi, color, label in AQI_BANDS:
        if aqi <= hi:
            return color, label
    return "#7e0023", "Severe"


def draw_gauge(aqi, max_aqi=500):
    """Draw an SVG-based AQI speedometer gauge."""
    import math
    aqi = min(aqi, max_aqi)
    pct = aqi / max_aqi
    angle = -180 + pct * 180  # -180 (left) to 0 (right)
    rad = math.radians(angle)

    color, category = aqi_color_category(aqi)

    # Needle tip
    nx = 150 + 90 * math.cos(rad)
    ny = 150 + 90 * math.sin(rad)

    # Build arc segments
    band_colors = ["#00e400", "#ffff00", "#ff7e00", "#ff0000", "#8f3f97", "#7e0023"]
    segments = []
    for i, bc in enumerate(band_colors):
        sa = math.radians(-180 + i * 30)
        ea = math.radians(-180 + (i + 1) * 30)
        x1 = 150 + 110 * math.cos(sa)
        y1 = 150 + 110 * math.sin(sa)
        x2 = 150 + 110 * math.cos(ea)
        y2 = 150 + 110 * math.sin(ea)
        x3 = 150 + 80 * math.cos(ea)
        y3 = 150 + 80 * math.sin(ea)
        x4 = 150 + 80 * math.cos(sa)
        y4 = 150 + 80 * math.sin(sa)
        segments.append(
            f'<path d="M{x1:.1f},{y1:.1f} A110,110 0 0,1 {x2:.1f},{y2:.1f} '
            f'L{x3:.1f},{y3:.1f} A80,80 0 0,0 {x4:.1f},{y4:.1f} Z" fill="{bc}" opacity="0.9"/>'
        )

    svg = f"""
    <svg width="100%" height="auto" style="max-width: 300px;" viewBox="0 0 300 190" xmlns="http://www.w3.org/2000/svg">
      <rect width="300" height="190" rx="16" fill="#1a1a2e"/>
      {''.join(segments)}
      <!-- Needle -->
      <line x1="150" y1="150" x2="{nx:.1f}" y2="{ny:.1f}"
            stroke="white" stroke-width="3" stroke-linecap="round"/>
      <circle cx="150" cy="150" r="8" fill="white"/>
      <!-- AQI value -->
      <text x="150" y="138" text-anchor="middle"
            font-family="Space Mono, monospace" font-size="22" font-weight="700"
            fill="{color}">{int(aqi)}</text>
      <text x="150" y="155" text-anchor="middle"
            font-family="DM Sans, sans-serif" font-size="11" fill="#8892b0">AQI</text>
      <!-- Labels -->
      <text x="38" y="155" font-size="9" fill="#555" font-family="monospace">0</text>
      <text x="260" y="155" font-size="9" fill="#555" font-family="monospace">500</text>
      <!-- Category badge -->
      <rect x="90" y="158" width="120" height="20" rx="10" fill="{color}22"/>
      <text x="150" y="172" text-anchor="middle"
            font-family="DM Sans, sans-serif" font-size="11" font-weight="600"
            fill="{color}">{category}</text>
    </svg>
    """
    return svg


def plot_trend(recent_vals, predicted_aqi):
    """7-day AQI trend + prediction chart."""
    fig, ax = plt.subplots(figsize=(7, 2.8))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#16213e')

    days = [f"Day -{6-i}" for i in range(len(recent_vals))]
    days.append("Tomorrow")
    vals = list(recent_vals) + [predicted_aqi]

    # Color each point
    colors = [aqi_color_category(v)[0] for v in vals]

    ax.plot(days[:-1], vals[:-1], color='#64ffda', linewidth=2, zorder=3)
    ax.plot([days[-2], days[-1]], [vals[-2], vals[-1]],
            color='#64ffda', linewidth=2, linestyle='--', zorder=3)

    for i, (x, y, c) in enumerate(zip(days, vals, colors)):
        ax.scatter(x, y, color=c, s=60, zorder=5, edgecolors='white', linewidths=0.8)

    #AQI band backgrounds
    for lo, hi, color, _ in AQI_BANDS:
        ax.axhspan(lo, min(hi, max(vals)*1.2), alpha=0.06, color=color)

    ax.set_xticks(days)
    ax.set_xticklabels(days, fontsize=8, color='#8892b0', rotation=20)
    ax.tick_params(axis='y', colors='#8892b0', labelsize=8)
    ax.set_ylabel("AQI", color='#8892b0', fontsize=9)
    ax.spines['bottom'].set_color('#0f3460')
    ax.spines['left'].set_color('#0f3460')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title("7-Day AQI Trend + Forecast", color='#ccd6f6', fontsize=10, pad=10)

    #Prediction annotation
    ax.annotate(f"Predicted\n{predicted_aqi:.0f}",
                xy=(days[-1], predicted_aqi),
                xytext=(-40, 20), textcoords='offset points',
                color='#64ffda', fontsize=8,
                arrowprops=dict(arrowstyle='->', color='#64ffda', lw=1))

    plt.tight_layout()
    return fig


def plot_pollutants(city):
    """Pollutant breakdown bar chart using latest available data."""
    try:
        df = load_city_data(city)
        df = clean_dataframe(df)
        pollutants = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
        available  = [p for p in pollutants if p in df.columns]
        if not available:
            return None
        latest = df[available].dropna().iloc[-1]

        fig, ax = plt.subplots(figsize=(7, 2.8))
        fig.patch.set_facecolor('#1a1a2e')
        ax.set_facecolor('#16213e')

        bar_colors = ['#64ffda', '#00b4d8', '#0077b6', '#f77f00', '#d62828', '#7209b7']
        bars = ax.bar(available, latest.values,
                      color=bar_colors[:len(available)], width=0.5, zorder=3)

        for bar, val in zip(bars, latest.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{val:.1f}', ha='center', va='bottom',
                    color='#ccd6f6', fontsize=8)

        ax.set_ylabel("µg/m³", color='#8892b0', fontsize=9)
        ax.tick_params(colors='#8892b0', labelsize=8)
        ax.set_title("Latest Pollutant Levels", color='#ccd6f6', fontsize=10, pad=10)
        ax.spines['bottom'].set_color('#0f3460')
        ax.spines['left'].set_color('#0f3460')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylim(0, latest.values.max() * 1.3)
        ax.yaxis.grid(True, color='#0f3460', linewidth=0.5)
        ax.set_axisbelow(True)

        plt.tight_layout()
        return fig
    except Exception:
        return None


#Loading model
@st.cache_resource
def load_model_artifacts():
    return load_artifacts()

try:
    model, feature_scaler, target_scaler = load_model_artifacts()
except Exception as e:
    st.error(f" Model not found. Run `python train.py --city Delhi` first.\n\n{e}")
    st.stop()

#Header
st.markdown("""
<div style="text-align:center; padding: 10px 0 24px 0;">
   <span style="font-family:'Space Mono',monospace; font-size:2.2rem;
             font-weight:700; color:#64ffda; letter-spacing:2px;">AIRGUARD</span>
    <p style="color:#8892b0; font-size:0.95rem; margin-top:4px;">
        Personalized AQI Health Risk Forecaster · Powered by Bidirectional LSTM
    </p>
</div>
""", unsafe_allow_html=True)

#Sidebar 
with st.sidebar:
    st.markdown('<div class="section-header">Your Profile</div>', unsafe_allow_html=True)

    cities = get_available_cities()
    city = st.selectbox(
        " City",
        options=cities,
        index=cities.index("Delhi") if "Delhi" in cities else 0
    )

    date_info = get_city_date_range(city)
    if date_info:
        st.caption(f" Data: {date_info['start']} → {date_info['end']} ({date_info['rows']} days)")

    st.divider()

    age = st.slider(" Age", 1, 100, 25)
    condition = st.selectbox(
        "Health Condition",
        ["healthy", "asthmatic", "diabetic", "elderly"],
        format_func=lambda x: x.capitalize()
    )

    st.divider()
    predict_btn = st.button(" Forecast AQI", use_container_width=True, type="primary")

    st.markdown("""
    <div style="margin-top:24px; padding:12px; background:#0f3460;
                border-radius:10px; font-size:0.78rem; color:#8892b0;">
        <b style="color:#64ffda;">How it works</b><br><br>
        A BiLSTM model trained on India AQI 2015–2020 data predicts
        next-day AQI using rolling pollutant statistics, lag features,
        and cyclical time encodings.
    </div>
    """, unsafe_allow_html=True)

#Main content
if not predict_btn:
    #Landing state
    st.markdown("""
    <div style="text-align:center; padding:60px 20px; color:#4a5568;">
        <div style="font-size:3rem;"></div>
        <p style="font-size:1.1rem; color:#8892b0; margin-top:12px;">
            Select your city and profile in the sidebar, then click
            <b style="color:#64ffda;">Forecast AQI</b> to get your personalized health forecast.
        </p>
    </div>
    """, unsafe_allow_html=True)

    #AQI legend
  
st.markdown('<div class="section-header">AQI Scale Reference</div>', unsafe_allow_html=True)

cols = st.columns(6)
uniform_color = "#64ffda"

for i, (lo, hi, _, label) in enumerate(AQI_BANDS):
            with cols[i]:
                st.markdown(f"""
                <div style="background:{uniform_color}22; border:1px solid {uniform_color};
                            border-radius:10px; padding:10px; text-align:center;">
                    <div style="color:{uniform_color}; font-weight:600; font-size:0.85rem;">{label}</div>
                    <div style="color:#8892b0; font-size:0.75rem;">{lo}–{hi}</div>
                </div>
                """, unsafe_allow_html=True)
else:
    #Run inference 
    with st.spinner(f"Forecasting AQI for {city}..."):
        try:
            inference_result = predict_next_day(city, model, feature_scaler, target_scaler)
            predicted_aqi    = inference_result["predicted_aqi"]
            recent_aqi_vals  = inference_result["recent_aqi_vals"]
            result = get_advice(predicted_aqi, age, condition, recent_aqi_vals)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

    color, category = aqi_color_category(predicted_aqi)
    confidence_lo = max(0, predicted_aqi - 15)
    confidence_hi = predicted_aqi + 15

    #Row 1:Gauge+ Metrics
    col_gauge, col_metrics = st.columns([1, 2])

    with col_gauge:
        st.markdown('<div class="section-header">AQI Forecast</div>', unsafe_allow_html=True)
        st.markdown(draw_gauge(predicted_aqi), unsafe_allow_html=True)
        st.markdown(f"""
        <div style="text-align:center; margin-top:6px;">
            <span style="font-size:0.78rem; color:#8892b0; font-family:'Space Mono',monospace;">
                Confidence range: {confidence_lo:.0f} – {confidence_hi:.0f}
            </span>
        </div>
        """, unsafe_allow_html=True)

    with col_metrics:
        st.markdown('<div class="section-header">Summary</div>', unsafe_allow_html=True)
        m1, m2 = st.columns(2)
        m3, m4 = st.columns(2)

        trend_icons = {"rising": "⬆️", "falling": "⬇️", "stable": "➡️"}
        trend_icon  = trend_icons.get(result.get("trend", "stable"), "➡️")

        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Predicted AQI</div>
                <div class="metric-value" style="color:{color};">{predicted_aqi}</div>
                <div class="metric-sub">{category}</div>
            </div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Risk Level</div>
                <div class="metric-value">{result.get('risk_level','—')}</div>
                <div class="metric-sub">For your profile</div>
            </div>""", unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">AQI Trend</div>
                <div class="metric-value">{trend_icon}</div>
                <div class="metric-sub">{result.get('trend','stable').capitalize()}</div>
            </div>""", unsafe_allow_html=True)
        with m4:
            recent_avg = round(float(np.mean(recent_aqi_vals)), 1) if recent_aqi_vals else "—"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">7-Day Avg AQI</div>
                <div class="metric-value">{recent_avg}</div>
                <div class="metric-sub">Historical</div>
            </div>""", unsafe_allow_html=True)

        #Profile chips
        st.markdown(f"""
        <div style="margin-top:8px;">
            <span class="profile-chip"> {city}</span>
            <span class="profile-chip"> Age {age}</span>
            <span class="profile-chip"> {condition.capitalize()}</span>
            <span class="profile-chip">{' Vulnerable' if result.get('vulnerable') else ' Standard'}</span>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    #Row 2: Charts
    col_trend, col_poll = st.columns(2)

    with col_trend:
        st.markdown('<div class="section-header">AQI Trend & Forecast</div>', unsafe_allow_html=True)
        if recent_aqi_vals:
            fig_trend = plot_trend(recent_aqi_vals, predicted_aqi)
            st.pyplot(fig_trend, use_container_width=True)
            plt.close(fig_trend)
        else:
            st.info("No recent AQI history available.")

    with col_poll:
        st.markdown('<div class="section-header">Pollutant Breakdown</div>', unsafe_allow_html=True)
        fig_poll = plot_pollutants(city)
        if fig_poll:
            st.pyplot(fig_poll, use_container_width=True)
            plt.close(fig_poll)
        else:
            st.info("Pollutant data not available for this city.")

    st.divider()

    #Row 3: Health Advisory
    st.markdown('<div class="section-header">Health Advisory</div>', unsafe_allow_html=True)

    adv_col, action_col = st.columns([3, 2])

    with adv_col:
        risk_color = result.get("color", color)
        st.markdown(f"""
        <div class="advisory-card" style="background:{risk_color}12;
             border-left-color:{risk_color};">
            <div style="font-size:0.72rem; text-transform:uppercase; letter-spacing:2px;
                        color:{risk_color}; margin-bottom:8px; font-family:'Space Mono',monospace;">
                Personal Advisory
            </div>
            <p style="margin:0; color:#ccd6f6; font-size:1rem; line-height:1.6;">
                {result.get('advice', '—')}
            </p>
        </div>
        """, unsafe_allow_html=True)

        if result.get("trend_note"):
            st.info(f" {result['trend_note']}")
        if result.get("boundary_caveat"):
            st.warning(f" {result['boundary_caveat']}")
        if result.get("escalated"):
            st.warning("Risk escalated — AQI is rising near a category boundary.")

    with action_col:
        st.markdown(f"""
        <div style="background:#0f3460; border-radius:14px; padding:18px;">
            <div style="font-size:0.72rem; text-transform:uppercase; letter-spacing:2px;
                        color:#64ffda; margin-bottom:12px; font-family:'Space Mono',monospace;">
                Recommended Actions
            </div>
        """, unsafe_allow_html=True)
        for action in result.get("actions", []):
            st.markdown(f" {action}")
        st.markdown("</div>", unsafe_allow_html=True)

    st.divider()

    #Row 4: Demographic Risk Cards
    st.markdown('<div class="section-header">Demographic Risk Guide</div>', unsafe_allow_html=True)

    groups = [
        ("Children", "Avoid outdoor play. Keep windows closed.", predicted_aqi > 100),
        ("Elderly",  "Stay indoors. Use air purifier if available.", predicted_aqi > 150),
        ("Asthmatic","Carry inhaler. Avoid strenuous activity outdoors.", predicted_aqi > 50),
        ("Active",   "Reschedule outdoor workouts if AQI > 150.", predicted_aqi > 150),
    ]

    dcols = st.columns(4)
    for i, (group, tip, at_risk) in enumerate(groups):
        risk_c = "#ff4444" if at_risk else "#00e400"
        risk_t = "At Risk" if at_risk else "Safe"
        with dcols[i]:
            st.markdown(f"""
            <div style="background:#16213e; border:1px solid {risk_c}44;
                        border-radius:14px; padding:16px; text-align:center; height:160px;">
                <div style="font-size:1.6rem;">{group.split()[0]}</div>
                <div style="font-size:0.82rem; color:#ccd6f6; font-weight:600;
                            margin:6px 0 4px 0;">{' '.join(group.split(' ')[1:])}</div>
                <div style="font-size:0.72rem; color:#8892b0; margin-bottom:10px;
                            line-height:1.4;">{tip}</div>
                <span style="background:{risk_c}22; color:{risk_c}; border-radius:20px;
                             padding:2px 12px; font-size:0.72rem; font-weight:600;">
                    {risk_t}
                </span>
            </div>
            """, unsafe_allow_html=True)

    #Footer
    st.markdown(f"""
    <div class="source-note">
        AirGuard · BiLSTM Model · India AQI Dataset (Kaggle) · 2015–2020 ·
        Forecast generated {datetime.now().strftime('%d %b %Y, %H:%M')}
    </div>
    """, unsafe_allow_html=True)