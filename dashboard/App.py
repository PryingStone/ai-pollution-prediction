"""
dashboard/app.py
----------------
Part of: AirSenseAI (AI-based Real-Time Pollution Prediction and Alert System)

PURPOSE
    Streamlit dashboard that displays:
        - Current live AQI for Bengaluru (from WAQI API)
        - Model's 24-hour AQI prediction
        - AQI change and pollution trend (Improving / Stable / Worsening)
        - Color-coded risk category cards
        - Health recommendations for now and 24 hours ahead
        - A refresh button to re-fetch live data

HOW TO RUN
    From the project root folder, run:

        streamlit run dashboard/app.py

    Make sure your WAQI API key is set as an environment variable first:

        On Mac/Linux:   export WAQI_API_KEY="your_token_here"
        On Windows:     set WAQI_API_KEY=your_token_here

    The dashboard automatically imports run_prediction() from src/predict.py.
    Make sure you have run train.py first so the model files exist.
"""

import sys
import os

import streamlit as st

# ---------------------------------------------------------------------------
# PATH SETUP
# ---------------------------------------------------------------------------
# This file lives in dashboard/. We need Python to also look in src/ so it
# can find predict.py. This adds the project root to Python's search path.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from predict import run_prediction  # noqa: E402  (import after path setup)


# ---------------------------------------------------------------------------
# 1. PAGE CONFIG
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AirSenseAI – Bengaluru AQI Forecast",
    page_icon="🌿",
    layout="centered",
)


# ---------------------------------------------------------------------------
# 2. STYLING
# ---------------------------------------------------------------------------
# We inject a small block of CSS to color the AQI cards and clean up
# the default Streamlit look. Each AQI category gets its own color,
# matching the CPCB standard color scheme used on India's AQI boards.

AQI_COLORS = {
    "Good":        {"bg": "#55a84f", "text": "#ffffff"},   # green
    "Satisfactory":{"bg": "#a3c853", "text": "#1a1a1a"},   # yellow-green
    "Moderate":    {"bg": "#fff833", "text": "#1a1a1a"},   # yellow
    "Poor":        {"bg": "#f29c33", "text": "#1a1a1a"},   # orange
    "Very Poor":   {"bg": "#e93f33", "text": "#ffffff"},   # red
    "Severe":      {"bg": "#af2d24", "text": "#ffffff"},   # dark red
    "Unknown":     {"bg": "#888888", "text": "#ffffff"},   # grey fallback
}

st.markdown("""
<style>
    /* ── Page background and base font ── */
    .stApp { background-color: #f0f2f6; }

    /* ── AQI card ── */
    .aqi-card {
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 12px;
        text-align: center;
    }
    .aqi-card .label {
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        opacity: 0.85;
        margin-bottom: 4px;
    }
    .aqi-card .value {
        font-size: 3.2rem;
        font-weight: 800;
        line-height: 1.1;
    }
    .aqi-card .category {
        font-size: 1.05rem;
        font-weight: 600;
        margin-top: 4px;
    }

    /* ── Trend badge ── */
    .trend-badge {
        display: inline-block;
        border-radius: 20px;
        padding: 6px 18px;
        font-size: 0.95rem;
        font-weight: 700;
        margin: 0 auto;
    }

    /* ── Change chip ── */
    .change-chip {
        border-radius: 8px;
        padding: 16px 20px;
        text-align: center;
        background: #ffffff;
        border: 1px solid #dde1ea;
        margin-bottom: 12px;
    }
    .change-chip .label {
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.07em;
        text-transform: uppercase;
        color: #666;
        margin-bottom: 4px;
    }
    .change-chip .val {
        font-size: 2rem;
        font-weight: 800;
        color: #1a1a2e;
    }

    /* ── Advice box ── */
    .advice-box {
        background: #ffffff;
        border-left: 5px solid #4a90d9;
        border-radius: 8px;
        padding: 14px 18px;
        margin-top: 6px;
        font-size: 0.95rem;
        color: #2d2d2d;
        line-height: 1.55;
    }

    /* ── Section header ── */
    .section-header {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #888;
        margin: 24px 0 8px 0;
    }

    /* ── Error box ── */
    .error-box {
        background: #fff0f0;
        border: 1px solid #f5c6c6;
        border-radius: 10px;
        padding: 18px 22px;
        color: #8b0000;
        font-size: 0.95rem;
        line-height: 1.6;
    }

    /* ── Timestamp ── */
    .timestamp {
        font-size: 0.78rem;
        color: #999;
        text-align: center;
        margin-top: 4px;
    }

    /* Hide Streamlit's default top padding */
    .block-container { padding-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 3. HELPERS
# ---------------------------------------------------------------------------
def aqi_card_html(label, aqi_value, category):
    """
    Returns the HTML for a color-coded AQI display card.
    The background color is chosen from AQI_COLORS based on category.
    """
    colors = AQI_COLORS.get(category, AQI_COLORS["Unknown"])
    bg    = colors["bg"]
    fg    = colors["text"]
    return f"""
    <div class="aqi-card" style="background:{bg}; color:{fg};">
        <div class="label">{label}</div>
        <div class="value">{aqi_value}</div>
        <div class="category">{category}</div>
    </div>
    """


def trend_label(aqi_change):
    """
    Converts a numeric AQI change into a human-readable trend label.

    Thresholds:
        Change ≤ -10  →  Improving  (AQI going down = cleaner air)
        Change within ±10 →  Stable
        Change ≥ +10  →  Worsening
    """
    if aqi_change <= -10:
        return "Improving", "🟢", "#d4edda", "#1a5c2e"
    elif aqi_change >= 10:
        return "Worsening", "🔴", "#f8d7da", "#6b1a1a"
    else:
        return "Stable", "🟡", "#fff3cd", "#5a4600"


def change_display(aqi_change):
    """Returns a sign-prefixed string for the AQI change value."""
    if aqi_change > 0:
        return f"+{aqi_change}"
    return str(aqi_change)


# ---------------------------------------------------------------------------
# 4. FETCH DATA (cached per session, refreshed on button click)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_prediction(_refresh_key):
    """
    Calls run_prediction() from predict.py and returns the result dict.

    The _refresh_key parameter is a trick: when its value changes
    (which happens when the user clicks Refresh), Streamlit treats
    this as a new call and re-runs the function instead of returning
    the cached result.

    The underscore prefix tells Streamlit not to hash this argument
    itself, just use it as a cache-bust signal.
    """
    return run_prediction()


# ---------------------------------------------------------------------------
# 5. DASHBOARD LAYOUT
# ---------------------------------------------------------------------------
# ── Header ──────────────────────────────────────────────────────────────────
st.markdown("## 🌿 AirSenseAI")
st.markdown("**24-hour AQI forecast for Bengaluru** — powered by a Random Forest model trained on CPCB data.")

st.divider()

# ── Refresh button ──────────────────────────────────────────────────────────
# We store a counter in session_state. Each click increments it, which
# changes _refresh_key and forces get_prediction() to re-run.
if "refresh_count" not in st.session_state:
    st.session_state.refresh_count = 0

col_btn, col_spacer = st.columns([1, 4])
with col_btn:
    if st.button("🔄 Refresh", use_container_width=True):
        st.session_state.refresh_count += 1
        st.cache_data.clear()   # also clears the cache so the API is re-called

# ── Fetch data ───────────────────────────────────────────────────────────────
with st.spinner("Fetching live AQI and running prediction..."):
    try:
        result = get_prediction(st.session_state.refresh_count)
        fetch_error = None
    except FileNotFoundError as e:
        result = None
        fetch_error = (
            f"**Model files not found.**\n\n{e}\n\n"
            "Make sure you have run `preprocess.py` and then `train.py` first."
        )
    except ValueError as e:
        result = None
        fetch_error = (
            f"**API key or data problem.**\n\n{e}"
        )
    except (ConnectionError, TimeoutError) as e:
        result = None
        fetch_error = (
            f"**Could not reach the WAQI API.**\n\n{e}\n\n"
            "Check your internet connection and try refreshing."
        )
    except RuntimeError as e:
        result = None
        fetch_error = (
            f"**WAQI API error.**\n\n{e}\n\n"
            "The API may be temporarily unavailable. Try again in a moment."
        )
    except Exception as e:
        result = None
        fetch_error = (
            f"**Unexpected error.**\n\n{type(e).__name__}: {e}\n\n"
            "Check the terminal for a full traceback."
        )

# ── Error state ──────────────────────────────────────────────────────────────
if fetch_error:
    st.markdown(
        f'<div class="error-box">⚠️ {fetch_error}</div>',
        unsafe_allow_html=True,
    )
    st.stop()   # nothing more to render

# ── AQI Cards ────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Air Quality Index</div>', unsafe_allow_html=True)

col_now, col_pred = st.columns(2)

with col_now:
    st.markdown(
        aqi_card_html("Right now", result["current_aqi"], result["current_category"]),
        unsafe_allow_html=True,
    )

with col_pred:
    st.markdown(
        aqi_card_html("In 24 hours (predicted)", result["predicted_aqi"], result["predicted_category"]),
        unsafe_allow_html=True,
    )

# ── Change & Trend ───────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Forecast Summary</div>', unsafe_allow_html=True)

col_change, col_trend = st.columns(2)

with col_change:
    st.markdown(f"""
    <div class="change-chip">
        <div class="label">AQI Change</div>
        <div class="val">{change_display(result["aqi_change"])}</div>
    </div>
    """, unsafe_allow_html=True)

with col_trend:
    trend_text, trend_icon, trend_bg, trend_fg = trend_label(result["aqi_change"])
    st.markdown(f"""
    <div class="change-chip" style="background:{trend_bg}; border-color:{trend_bg};">
        <div class="label" style="color:{trend_fg};">Pollution Trend</div>
        <div class="val" style="color:{trend_fg}; font-size:1.5rem;">
            {trend_icon} {trend_text}
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Health Recommendations ────────────────────────────────────────────────────
st.markdown('<div class="section-header">Health Recommendations</div>', unsafe_allow_html=True)

st.markdown(f"""
<div class="advice-box">
    <strong>Now ({result["current_category"]}):</strong><br>
    {result["current_advice"]}
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="advice-box" style="border-left-color: #e07b39; margin-top: 10px;">
    <strong>In 24 hours ({result["predicted_category"]}):</strong><br>
    {result["predicted_advice"]}
</div>
""", unsafe_allow_html=True)

# ── Timestamps ────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="timestamp">
    AQI reading: {result["reading_time"]} &nbsp;·&nbsp;
    Prediction made: {result["prediction_time"]}
</div>
""", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "AirSenseAI · Data: WAQI API · Model: Random Forest trained on CPCB Bengaluru data (2015–2020) · "
    "AQI categories follow India CPCB standard."
)