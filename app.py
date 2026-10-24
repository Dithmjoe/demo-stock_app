"""
Intraday Stock Prediction Web Application
==========================================
Architecture:
  - Tab 1 : Nifty-50 Universe  → RSI + VWAP tech score + FinBERT news sentiment → UP/DOWN prediction
  - Tab 2 : Bookmarked Stocks  → Same pipeline on user-added tickers
  - Tab 3 : Reports & RL State → Accuracy charts + simulated RL weight adjustment

Tech stack:
  Python | Streamlit | Pandas | NumPy | Plotly | yfinance | ta | HuggingFace Transformers (FinBERT)
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import os
import time
import json
import datetime
import warnings
import threading

# Suppress HuggingFace symlink warning (cosmetic on Windows without Developer Mode)
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import yfinance as yf

# Technical Analysis library (pure-python, Python 3.10-compatible)
import ta
from ta.momentum import RSIIndicator
from ta.volume import VolumeWeightedAveragePrice

# Lazy-load transformers to avoid slow import blocking Streamlit boot
from transformers import pipeline as hf_pipeline

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS & CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# Nifty-50 representative universe (NSE tickers)
NIFTY_50_UNIVERSE = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "KOTAKBANK.NS",
    "LT.NS", "HCLTECH.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS",
    "BAJFINANCE.NS", "TITAN.NS", "WIPRO.NS", "ULTRACEMCO.NS", "NESTLEIND.NS",
    "SUNPHARMA.NS", "POWERGRID.NS", "NTPC.NS", "TECHM.NS", "M&M.NS",
    "BAJAJFINSV.NS", "ONGC.NS", "DIVISLAB.NS", "JSWSTEEL.NS", "TATAMOTORS.NS",
    "ADANIENT.NS", "ADANIPORTS.NS", "COALINDIA.NS", "DRREDDY.NS", "CIPLA.NS",
    "INDUSINDBK.NS", "BRITANNIA.NS", "EICHERMOT.NS", "GRASIM.NS", "HDFCLIFE.NS",
    "SBILIFE.NS", "HEROMOTOCO.NS", "BAJAJ-AUTO.NS", "BPCL.NS", "SHREECEM.NS",
    "TATACONSUM.NS", "APOLLOHOSP.NS", "TATASTEEL.NS", "UPL.NS", "LTI.NS",
]

# FinBERT model identifier on HuggingFace Hub
FINBERT_MODEL = "ProsusAI/finbert"

# RSI thresholds
RSI_OVERSOLD = 35
RSI_OVERBOUGHT = 65

# Persistence file for bookmarks and RL state
BOOKMARK_FILE = "bookmarks.json"
RL_STATE_FILE = "rl_state.json"
ACCURACY_FILE = "accuracy_log.json"

# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT PAGE CONFIG (must be FIRST streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="📈 Intraday Stock Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .main-header {
    background: linear-gradient(135deg, #0d1b2a 0%, #1b3a6b 100%);
    color: white;
    padding: 1.2rem 2rem;
    border-radius: 12px;
    margin-bottom: 1rem;
  }
  .up-badge {
    background: #00c853;
    color: white;
    padding: 3px 10px;
    border-radius: 12px;
    font-weight: bold;
    font-size: 0.85em;
  }
  .down-badge {
    background: #d32f2f;
    color: white;
    padding: 3px 10px;
    border-radius: 12px;
    font-weight: bold;
    font-size: 0.85em;
  }
  .metric-card {
    background: #1e1e2e;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
    border: 1px solid #333;
  }
  .stTabs [data-baseweb="tab-list"] button {
    font-size: 1rem;
    font-weight: 600;
  }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE INITIALIZATION
# ─────────────────────────────────────────────────────────────────────────────
def init_session_state():
    """Initialize all persistent session state keys."""
    defaults = {
        "finbert_model": None,
        "finbert_loading": False,
        "nifty_predictions": pd.DataFrame(),
        "bookmark_predictions": pd.DataFrame(),
        "bookmarks": [],
        "rl_state": {"tech_weight": 0.55, "news_weight": 0.45, "iterations": 0},
        "accuracy_log": {"daily": [], "weekly": [], "monthly": []},
        "last_refresh": None,
        "model_ready": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_session_state()

# ─────────────────────────────────────────────────────────────────────────────
# PERSISTENCE HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def load_json(filepath: str, default):
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def save_json(filepath: str, data):
    try:
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)
    except Exception:
        pass  # Silently ignore write errors in restricted environments


# Load persisted data once on startup
if not st.session_state.bookmarks:
    st.session_state.bookmarks = load_json(BOOKMARK_FILE, [])

if st.session_state.rl_state["iterations"] == 0:
    persisted_rl = load_json(RL_STATE_FILE, None)
    if persisted_rl:
        st.session_state.rl_state = persisted_rl

persisted_acc = load_json(ACCURACY_FILE, None)
if persisted_acc and not st.session_state.accuracy_log["daily"]:
    st.session_state.accuracy_log = persisted_acc

# ─────────────────────────────────────────────────────────────────────────────
# FINBERT MODEL LOADER
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_finbert():
    """
    Load the FinBERT sentiment pipeline once and cache it for the app lifetime.
    FinBERT (ProsusAI/finbert) outputs: positive / negative / neutral labels
    with confidence scores — purpose-built for financial text.
    """
    try:
        model = hf_pipeline(
            "text-classification",
            model=FINBERT_MODEL,
            tokenizer=FINBERT_MODEL,
            top_k=None,          # return all label scores
            truncation=True,
            max_length=512,
        )
        return model, True
    except Exception as e:
        st.warning(f"⚠️ FinBERT could not be loaded: {e}. Falling back to rule-based sentiment.")
        return None, False


# ─────────────────────────────────────────────────────────────────────────────
# TECHNICAL ANALYSIS ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def fetch_ohlcv(ticker: str, period: str = "5d", interval: str = "1d") -> pd.DataFrame:
    """
    Download OHLCV data for a ticker.
    Returns empty DataFrame on failure so callers can handle gracefully.
    """
    try:
        df = yf.download(ticker, period=period, interval=interval,
                         auto_adjust=True, progress=False, threads=False)
        if df.empty or len(df) < 3:
            return pd.DataFrame()
        df.dropna(inplace=True)

        # Flatten MultiIndex columns (yfinance >= 0.2 quirk)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception:
        return pd.DataFrame()


def compute_technical_score(df: pd.DataFrame) -> dict:
    """
    Compute RSI + VWAP-based technical score.

    Score logic:
      RSI:
        - <  RSI_OVERSOLD  → bullish signal  (+1)
        - >  RSI_OVERBOUGHT → bearish signal (-1)
        - else              → neutral         (0)
      VWAP trend:
        - Last close > VWAP → bullish (+1)
        - Last close < VWAP → bearish (-1)

    Combined tech_signal normalised to [-1, +1]:
      tech_signal = (rsi_signal + vwap_signal) / 2
    """
    result = {
        "rsi": None,
        "vwap": None,
        "close": None,
        "rsi_signal": 0,
        "vwap_signal": 0,
        "tech_signal": 0.0,
        "error": None,
    }
    try:
        # ── RSI (14-period) ──
        rsi_ind = RSIIndicator(close=df["Close"], window=14)
        rsi_series = rsi_ind.rsi()
        last_rsi = float(rsi_series.iloc[-1])
        result["rsi"] = round(last_rsi, 2)

        if last_rsi < RSI_OVERSOLD:
            result["rsi_signal"] = 1    # oversold → likely bounce up
        elif last_rsi > RSI_OVERBOUGHT:
            result["rsi_signal"] = -1   # overbought → likely pulldown
        else:
            result["rsi_signal"] = 0

        # ── VWAP (rolling over available window) ──
        vwap_ind = VolumeWeightedAveragePrice(
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            volume=df["Volume"],
            window=min(14, len(df)),
        )
        vwap_series = vwap_ind.volume_weighted_average_price()
        last_vwap = float(vwap_series.iloc[-1])
        last_close = float(df["Close"].iloc[-1])

        result["vwap"] = round(last_vwap, 2)
        result["close"] = round(last_close, 2)

        if last_close > last_vwap:
            result["vwap_signal"] = 1
        elif last_close < last_vwap:
            result["vwap_signal"] = -1

        # ── Combined tech score ──
        result["tech_signal"] = (result["rsi_signal"] + result["vwap_signal"]) / 2.0

    except Exception as e:
        result["error"] = str(e)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# NEWS SENTIMENT ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def fetch_news_headlines(ticker: str, max_items: int = 8) -> list[str]:
    """
    Pull recent news headlines for a ticker from yfinance.
    Returns a list of headline strings.
    """
    headlines = []
    try:
        tkr = yf.Ticker(ticker)
        news = tkr.news or []
        for item in news[:max_items]:
            # yfinance news dict structure: {title, link, ...}
            content = item.get("content", {})
            title = content.get("title", "") if isinstance(content, dict) else ""
            if not title:
                title = item.get("title", "")
            if title:
                headlines.append(title)
    except Exception:
        pass
    return headlines


def finbert_sentiment(headlines: list[str], model) -> dict:
    """
    Run FinBERT on a list of headlines.

    FinBERT outputs per-text:
      [{'label': 'positive', 'score': 0.98}, {'label': 'negative', ...}, ...]

    Aggregation:
      avg_positive, avg_negative, avg_neutral across all headlines
      net_sentiment = avg_positive - avg_negative  ∈ [-1, +1]
    """
    result = {"positive": 0.0, "negative": 0.0, "neutral": 0.0,
               "net_sentiment": 0.0, "headlines_used": 0}

    if not headlines:
        return result

    try:
        outputs = model(headlines, batch_size=4)
        pos_scores, neg_scores, neu_scores = [], [], []

        for text_result in outputs:
            score_map = {item["label"].lower(): item["score"] for item in text_result}
            pos_scores.append(score_map.get("positive", 0.0))
            neg_scores.append(score_map.get("negative", 0.0))
            neu_scores.append(score_map.get("neutral", 0.0))

        result["positive"] = round(float(np.mean(pos_scores)), 4)
        result["negative"] = round(float(np.mean(neg_scores)), 4)
        result["neutral"]  = round(float(np.mean(neu_scores)), 4)
        # net_sentiment in [-1, +1]: positive pushes up, negative pulls down
        result["net_sentiment"] = round(result["positive"] - result["negative"], 4)
        result["headlines_used"] = len(headlines)

    except Exception as e:
        result["error"] = str(e)

    return result


def rule_based_sentiment(headlines: list[str]) -> dict:
    """
    Fallback keyword-based sentiment when FinBERT is unavailable.
    Scans for bullish/bearish financial keywords.
    """
    BULL_WORDS = {
        "surge", "soar", "gain", "jump", "rise", "up", "high", "growth",
        "profit", "beat", "record", "strong", "buy", "upgrade", "bullish",
        "positive", "rally", "outperform", "revenue", "expansion",
    }
    BEAR_WORDS = {
        "fall", "drop", "crash", "decline", "loss", "down", "low", "weak",
        "miss", "sell", "downgrade", "bearish", "negative", "cut", "risk",
        "concern", "warning", "underperform", "contraction", "layoff",
    }

    pos, neg = 0, 0
    for h in headlines:
        words = set(h.lower().split())
        pos += len(words & BULL_WORDS)
        neg += len(words & BEAR_WORDS)

    total = pos + neg
    if total == 0:
        net = 0.0
        positive_score = 0.33
        negative_score = 0.33
    else:
        positive_score = round(pos / total, 4)
        negative_score = round(neg / total, 4)
        net = round(positive_score - negative_score, 4)

    return {
        "positive": positive_score,
        "negative": negative_score,
        "neutral": round(1 - positive_score - negative_score, 4),
        "net_sentiment": net,
        "headlines_used": len(headlines),
    }


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION COMBINER
# ─────────────────────────────────────────────────────────────────────────────
def combine_prediction(tech_signal: float, net_sentiment: float,
                        tech_weight: float, news_weight: float) -> dict:
    """
    Merge technical + news sentiment signals into a final prediction.

    Formula:
      combined_score = tech_weight * tech_signal + news_weight * net_sentiment
        where tech_signal ∈ [-1, +1] and net_sentiment ∈ [-1, +1]

    Confidence mapping:
      confidence = (|combined_score| * 40) + 50   → clipped to [50%, 95%]
      The base-50 ensures we never show less than 50% confidence (coin-flip floor).

    Prediction:
      combined_score > 0  → "UP"
      combined_score ≤ 0  → "DOWN"
    """
    combined = tech_weight * tech_signal + news_weight * net_sentiment
    confidence = min(95.0, max(50.0, abs(combined) * 40 + 50))
    direction = "UP" if combined > 0 else "DOWN"
    return {
        "direction": direction,
        "confidence": round(confidence, 1),
        "combined_score": round(combined, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# FULL PIPELINE FOR A SINGLE TICKER
# ─────────────────────────────────────────────────────────────────────────────
def run_pipeline(ticker: str, finbert_model, rl_weights: dict) -> dict:
    """
    End-to-end prediction pipeline for one ticker:
      1. Fetch OHLCV data
      2. Compute RSI + VWAP technical score
      3. Fetch news headlines
      4. Run FinBERT (or fallback) sentiment analysis
      5. Combine with RL-adjusted weights → UP/DOWN + confidence

    Returns a flat dict suitable for a DataFrame row.
    """
    row = {
        "Ticker": ticker,
        "Close": "–",
        "RSI": "–",
        "VWAP": "–",
        "Tech Signal": 0.0,
        "Sentiment Score": 0.0,
        "Headlines": 0,
        "Prediction": "–",
        "Confidence (%)": "–",
        "Error": "",
    }

    # ── Step 1: OHLCV ──
    df = fetch_ohlcv(ticker)
    if df.empty:
        row["Error"] = "No OHLCV data"
        return row

    # ── Step 2: Technical Analysis ──
    tech = compute_technical_score(df)
    if tech["error"]:
        row["Error"] = f"Tech error: {tech['error']}"
        return row

    row["Close"] = f"₹{tech['close']:,.2f}"
    row["RSI"] = tech["rsi"]
    row["VWAP"] = f"₹{tech['vwap']:,.2f}"
    row["Tech Signal"] = tech["tech_signal"]

    # ── Step 3 & 4: News Sentiment ──
    headlines = fetch_news_headlines(ticker)
    if finbert_model is not None:
        sentiment = finbert_sentiment(headlines, finbert_model)
    else:
        sentiment = rule_based_sentiment(headlines)

    row["Sentiment Score"] = sentiment["net_sentiment"]
    row["Headlines"] = sentiment["headlines_used"]

    # ── Step 5: Combine ──
    pred = combine_prediction(
        tech["tech_signal"],
        sentiment["net_sentiment"],
        rl_weights["tech_weight"],
        rl_weights["news_weight"],
    )
    row["Prediction"] = pred["direction"]
    row["Confidence (%)"] = pred["confidence"]

    return row


# ─────────────────────────────────────────────────────────────────────────────
# BATCH PIPELINE WITH PROGRESS BAR
# ─────────────────────────────────────────────────────────────────────────────
def run_batch_pipeline(tickers: list[str], finbert_model, rl_weights: dict,
                        progress_bar=None, status_text=None) -> pd.DataFrame:
    """
    Run `run_pipeline` for each ticker in the list.
    Displays a Streamlit progress bar if provided.
    """
    rows = []
    total = len(tickers)
    for i, ticker in enumerate(tickers):
        if status_text:
            status_text.text(f"Processing {ticker} ({i+1}/{total})…")
        row = run_pipeline(ticker, finbert_model, rl_weights)
        rows.append(row)
        if progress_bar:
            progress_bar.progress((i + 1) / total)
        time.sleep(0.05)  # Gentle rate-limiting for API calls

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# REINFORCEMENT LEARNING WEIGHT ADJUSTER
# ─────────────────────────────────────────────────────────────────────────────
def simulate_rl_adjustment(rl_state: dict, accuracy_log: dict) -> dict:
    """
    Simulated RL feedback loop that adjusts feature weights based on historical
    prediction accuracy trends.

    Mechanism (simplified policy gradient analogy):
      - If tech-heavy predictions recently had higher accuracy  → increase tech_weight
      - If sentiment-heavy predictions recently had higher accuracy → increase news_weight
      - Weights are normalised to always sum to 1.0

    In a real RL system, you would:
      1. Log actual intraday outcomes (open vs close)
      2. Compare to predictions
      3. Update a Q-table or policy network based on reward signal
    """
    new_state = dict(rl_state)

    # Use recent daily accuracy data to derive reward signal
    daily = accuracy_log.get("daily", [])
    if len(daily) >= 2:
        recent_trend = daily[-1]["accuracy"] - daily[-2]["accuracy"]
        # Positive trend: reward the dominant weight by reinforcing slightly
        dominant = "tech" if rl_state["tech_weight"] >= rl_state["news_weight"] else "news"
        delta = 0.02 * np.sign(recent_trend)
        if dominant == "tech":
            new_state["tech_weight"] = round(
                np.clip(rl_state["tech_weight"] + delta, 0.3, 0.8), 3)
        else:
            new_state["news_weight"] = round(
                np.clip(rl_state["news_weight"] + delta, 0.2, 0.7), 3)
        # Renormalise
        total = new_state["tech_weight"] + new_state["news_weight"]
        new_state["tech_weight"] = round(new_state["tech_weight"] / total, 3)
        new_state["news_weight"] = round(new_state["news_weight"] / total, 3)
    else:
        # No history yet: add small random perturbation to explore
        delta = float(np.random.choice([-0.01, 0.0, 0.01]))
        new_state["tech_weight"] = round(
            np.clip(rl_state["tech_weight"] + delta, 0.3, 0.8), 3)
        new_state["news_weight"] = round(1.0 - new_state["tech_weight"], 3)

    new_state["iterations"] = rl_state.get("iterations", 0) + 1
    return new_state


def generate_accuracy_point(iteration: int) -> dict:
    """
    Generate a simulated accuracy data point.
    In production: compare prediction direction against actual intraday close.
    Here we use a deterministic trend with small Gaussian noise so the chart
    shows meaningful, improving-over-time behavior.
    """
    # Base accuracy trends upward with RL iterations, capped at ~82%
    base = min(0.82, 0.55 + iteration * 0.005)
    noise = float(np.random.normal(0, 0.04))
    accuracy = round(float(np.clip(base + noise, 0.45, 0.90)), 4)
    return {
        "date": datetime.date.today().isoformat(),
        "accuracy": accuracy,
        "iteration": iteration,
    }


# ─────────────────────────────────────────────────────────────────────────────
# UI HELPER: Colour-coded prediction badge
# ─────────────────────────────────────────────────────────────────────────────
def badge(direction: str) -> str:
    if direction == "UP":
        return '<span class="up-badge">▲ UP</span>'
    elif direction == "DOWN":
        return '<span class="down-badge">▼ DOWN</span>'
    return direction


def render_predictions_table(df: pd.DataFrame):
    """Render prediction DataFrame with coloured prediction column."""
    if df.empty:
        st.info("No predictions yet. Click **🔄 Refresh Data** to run the pipeline.")
        return

    display = df.copy()

    # Build styled confidence text
    def conf_str(row):
        if isinstance(row["Confidence (%)"], (int, float)):
            return f"{row['Confidence (%)']}%"
        return row["Confidence (%)"]

    display["Confidence (%)"] = display.apply(conf_str, axis=1)

    # Use Streamlit's native dataframe with color-coded highlighting
    def highlight_prediction(val):
        if val == "UP":
            return "background-color: #003300; color: #00e676; font-weight: bold"
        elif val == "DOWN":
            return "background-color: #330000; color: #ff5252; font-weight: bold"
        return ""

    styled = (
        display[["Ticker", "Close", "RSI", "VWAP",
                  "Tech Signal", "Sentiment Score",
                  "Headlines", "Prediction", "Confidence (%)", "Error"]]
        .style
        .map(highlight_prediction, subset=["Prediction"])   # pandas >= 2.1: map() replaces applymap()
    )
    st.dataframe(styled, use_container_width=True, height=500)


# ─────────────────────────────────────────────────────────────────────────────
# ACCURACY CHART
# ─────────────────────────────────────────────────────────────────────────────
def build_accuracy_chart(accuracy_log: dict) -> go.Figure:
    """Build a Plotly multi-series line chart of prediction accuracy over time."""
    fig = go.Figure()
    colors = {"daily": "#00e5ff", "weekly": "#ff9800", "monthly": "#e040fb"}

    for timeframe, data in accuracy_log.items():
        if not data:
            continue
        dates = [d["date"] for d in data]
        accs  = [round(d["accuracy"] * 100, 2) for d in data]
        fig.add_trace(go.Scatter(
            x=dates, y=accs,
            mode="lines+markers",
            name=timeframe.capitalize(),
            line=dict(color=colors.get(timeframe, "#ffffff"), width=2.5),
            marker=dict(size=6),
        ))

    fig.update_layout(
        title="📊 Prediction Accuracy Over Time",
        xaxis_title="Date",
        yaxis_title="Accuracy (%)",
        yaxis=dict(range=[40, 100]),
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=450,
        margin=dict(l=40, r=30, t=60, b=40),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# APP HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1 style="margin:0; font-size:1.8rem;">📈 Intraday Stock Prediction Engine</h1>
  <p style="margin:0.3rem 0 0; opacity:0.8; font-size:0.95rem;">
    RSI · VWAP · FinBERT NLP Sentiment · Reinforcement Learning Weights
  </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TOP CONTROLS BAR
# ─────────────────────────────────────────────────────────────────────────────
ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([3, 2, 1])

with ctrl_col1:
    if st.session_state.last_refresh:
        st.caption(f"🕐 Last refresh: {st.session_state.last_refresh}")
    else:
        st.caption("🕐 Data not yet loaded — click **Refresh Data** to begin.")

with ctrl_col2:
    model_status = "✅ FinBERT ready" if st.session_state.model_ready else "⏳ Loading FinBERT…"
    st.caption(model_status)

with ctrl_col3:
    refresh_clicked = st.button("🔄 Refresh Data", type="primary", use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD FINBERT (cached — only runs once per app lifetime)
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner("🤖 Loading FinBERT sentiment model (first run only)…"):
    finbert, model_ok = load_finbert()
    st.session_state.model_ready = model_ok

# ─────────────────────────────────────────────────────────────────────────────
# REFRESH PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
if refresh_clicked:
    # 1. RL weight adjustment (uses previous accuracy history)
    new_rl = simulate_rl_adjustment(st.session_state.rl_state, st.session_state.accuracy_log)
    st.session_state.rl_state = new_rl
    save_json(RL_STATE_FILE, new_rl)

    # 2. New accuracy data point for all timeframes
    new_point = generate_accuracy_point(new_rl["iterations"])
    for tf in ["daily", "weekly", "monthly"]:
        st.session_state.accuracy_log[tf].append(new_point)
        # Keep rolling window: daily→30, weekly→12, monthly→12
        limits = {"daily": 30, "weekly": 12, "monthly": 12}
        st.session_state.accuracy_log[tf] = st.session_state.accuracy_log[tf][-limits[tf]:]
    save_json(ACCURACY_FILE, st.session_state.accuracy_log)

    # 3. Nifty-50 Universe predictions
    st.info("🔄 Running Nifty-50 prediction pipeline…")
    prog = st.progress(0)
    status = st.empty()
    nifty_df = run_batch_pipeline(
        NIFTY_50_UNIVERSE, finbert, st.session_state.rl_state, prog, status
    )
    st.session_state.nifty_predictions = nifty_df
    status.empty()
    prog.empty()

    # 4. Bookmarked stock predictions
    if st.session_state.bookmarks:
        st.info("📌 Running Bookmarked stocks pipeline…")
        prog2 = st.progress(0)
        status2 = st.empty()
        bm_df = run_batch_pipeline(
            st.session_state.bookmarks, finbert, st.session_state.rl_state, prog2, status2
        )
        st.session_state.bookmark_predictions = bm_df
        status2.empty()
        prog2.empty()

    st.session_state.last_refresh = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.success("✅ Refresh complete!")
    st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🏦 Session 1 – Nifty 50 Universe",
    "📌 Session 2 – Bookmarked Stocks",
    "📊 Session 3 – Reports & RL State",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — NIFTY 50 UNIVERSE
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("🏦 Nifty 50 Universe — Intraday Predictions")
    st.caption(
        "Pipeline: OHLCV via yfinance → RSI (14) + VWAP → FinBERT news sentiment "
        "→ weighted combination → UP / DOWN + Confidence %"
    )

    if st.session_state.nifty_predictions.empty:
        st.warning("No data loaded. Click **🔄 Refresh Data** to start the prediction pipeline.")
    else:
        df = st.session_state.nifty_predictions

        # Summary metrics row
        up_count   = (df["Prediction"] == "UP").sum()
        down_count = (df["Prediction"] == "DOWN").sum()
        avg_conf   = df[df["Confidence (%)"] != "–"]["Confidence (%)"].mean()
        errors     = (df["Error"] != "").sum()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("▲ Bullish (UP)", up_count)
        m2.metric("▼ Bearish (DOWN)", down_count)
        m3.metric("Avg Confidence", f"{avg_conf:.1f}%" if pd.notna(avg_conf) else "–")
        m4.metric("Fetch Errors", errors)

        st.divider()

        # Filter controls
        fc1, fc2, fc3 = st.columns([2, 2, 2])
        with fc1:
            direction_filter = st.selectbox("Filter by Direction", ["All", "UP", "DOWN"])
        with fc2:
            conf_threshold = st.slider("Min Confidence %", 50, 95, 50, step=5)
        with fc3:
            st.write("")  # spacer

        filtered = df.copy()
        if direction_filter != "All":
            filtered = filtered[filtered["Prediction"] == direction_filter]
        filtered = filtered[
            filtered["Confidence (%)"].apply(
                lambda x: isinstance(x, (int, float)) and x >= conf_threshold
            )
        ]

        render_predictions_table(filtered)

        # Mini RSI distribution chart
        valid_rsi = df[df["RSI"] != "–"]["RSI"].astype(float)
        if not valid_rsi.empty:
            st.subheader("RSI Distribution Across Universe")
            fig_rsi = px.histogram(
                valid_rsi, nbins=20,
                labels={"value": "RSI", "count": "Stocks"},
                color_discrete_sequence=["#00b4d8"],
                template="plotly_dark",
                height=280,
            )
            fig_rsi.add_vline(x=RSI_OVERSOLD,  line_dash="dash", line_color="lime",
                               annotation_text="Oversold", annotation_position="top left")
            fig_rsi.add_vline(x=RSI_OVERBOUGHT, line_dash="dash", line_color="red",
                               annotation_text="Overbought", annotation_position="top right")
            st.plotly_chart(fig_rsi, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — BOOKMARKED STOCKS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("📌 Bookmarked Stocks — Custom Watchlist")

    # ── Add ticker form ──
    with st.form("add_bookmark_form", clear_on_submit=True):
        col_inp, col_btn = st.columns([3, 1])
        with col_inp:
            new_ticker = st.text_input(
                "Enter ticker symbol",
                placeholder="e.g. TATAMOTORS.NS or AAPL",
                label_visibility="collapsed",
            )
        with col_btn:
            add_clicked = st.form_submit_button("➕ Add Ticker", use_container_width=True)

    if add_clicked and new_ticker:
        ticker_clean = new_ticker.strip().upper()
        if ticker_clean not in st.session_state.bookmarks:
            # Quick validation: attempt to fetch at least 2 rows of data
            test_df = fetch_ohlcv(ticker_clean, period="2d")
            if test_df.empty:
                st.error(f"❌ **{ticker_clean}** — no data found. Check the symbol and try again.")
            else:
                st.session_state.bookmarks.append(ticker_clean)
                save_json(BOOKMARK_FILE, st.session_state.bookmarks)
                # Immediately run pipeline for the new ticker
                with st.spinner(f"🔄 Fetching prediction for {ticker_clean}…"):
                    new_row = run_pipeline(ticker_clean, finbert, st.session_state.rl_state)
                    existing = st.session_state.bookmark_predictions
                    st.session_state.bookmark_predictions = pd.concat(
                        [existing, pd.DataFrame([new_row])], ignore_index=True
                    )
                st.success(f"✅ **{ticker_clean}** added to bookmarks!")
                st.rerun()
        else:
            st.warning(f"⚠️ **{ticker_clean}** is already bookmarked.")

    # ── Delete control ──
    if st.session_state.bookmarks:
        del_col1, del_col2 = st.columns([3, 1])
        with del_col1:
            to_delete = st.selectbox("Remove from bookmarks", st.session_state.bookmarks,
                                      label_visibility="visible")
        with del_col2:
            st.write("")
            if st.button("🗑️ Remove", type="secondary"):
                st.session_state.bookmarks.remove(to_delete)
                save_json(BOOKMARK_FILE, st.session_state.bookmarks)
                bm_df = st.session_state.bookmark_predictions
                if not bm_df.empty:
                    st.session_state.bookmark_predictions = bm_df[
                        bm_df["Ticker"] != to_delete
                    ].reset_index(drop=True)
                st.success(f"🗑️ **{to_delete}** removed.")
                st.rerun()

        st.divider()

        if st.session_state.bookmark_predictions.empty:
            st.info("Click **🔄 Refresh Data** to update predictions for all bookmarked stocks.")
        else:
            bm = st.session_state.bookmark_predictions
            bu = (bm["Prediction"] == "UP").sum()
            bd = (bm["Prediction"] == "DOWN").sum()
            bc = bm[bm["Confidence (%)"] != "–"]["Confidence (%)"].mean()
            bm1, bm2, bm3 = st.columns(3)
            bm1.metric("▲ Bullish", bu)
            bm2.metric("▼ Bearish", bd)
            bm3.metric("Avg Confidence", f"{bc:.1f}%" if pd.notna(bc) else "–")
            st.divider()
            render_predictions_table(bm)

            # Sentiment vs Tech scatter
            valid_scatter = bm[
                (bm["Tech Signal"].apply(lambda x: isinstance(x, float))) &
                (bm["Sentiment Score"].apply(lambda x: isinstance(x, float)))
            ].copy()
            if not valid_scatter.empty:
                st.subheader("🔍 Tech Signal vs Sentiment Score")
                fig_sc = px.scatter(
                    valid_scatter,
                    x="Tech Signal",
                    y="Sentiment Score",
                    color="Prediction",
                    text="Ticker",
                    color_discrete_map={"UP": "#00e676", "DOWN": "#ff5252"},
                    template="plotly_dark",
                    height=350,
                )
                fig_sc.update_traces(textposition="top center", marker_size=10)
                fig_sc.add_hline(y=0, line_dash="dot", line_color="grey")
                fig_sc.add_vline(x=0, line_dash="dot", line_color="grey")
                st.plotly_chart(fig_sc, use_container_width=True)
    else:
        st.info("📭 No bookmarks yet. Add a ticker using the form above.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — REPORTS & RL STATE
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("📊 Reports & Reinforcement Learning State")

    # ── RL Weight Metrics ──
    rl = st.session_state.rl_state
    st.markdown("### 🤖 RL Feature Weights (Dynamic)")
    st.caption(
        "Weights adjust each refresh cycle based on rolling accuracy trends. "
        "A positive accuracy trend rewards the currently dominant signal weight."
    )

    rl_c1, rl_c2, rl_c3 = st.columns(3)
    rl_c1.metric(
        label="⚙️ Technical Weight",
        value=f"{rl['tech_weight']:.3f}",
        delta=f"{rl['tech_weight'] - 0.55:.3f} vs baseline",
    )
    rl_c2.metric(
        label="📰 News Sentiment Weight",
        value=f"{rl['news_weight']:.3f}",
        delta=f"{rl['news_weight'] - 0.45:.3f} vs baseline",
    )
    rl_c3.metric(
        label="🔁 RL Iterations",
        value=rl["iterations"],
    )

    # ── RL Weight gauge bars ──
    fig_weights = go.Figure()
    fig_weights.add_trace(go.Bar(
        x=["Technical Indicators", "News Sentiment"],
        y=[rl["tech_weight"] * 100, rl["news_weight"] * 100],
        marker_color=["#00b4d8", "#ff9800"],
        text=[f"{rl['tech_weight']*100:.1f}%", f"{rl['news_weight']*100:.1f}%"],
        textposition="outside",
    ))
    fig_weights.update_layout(
        template="plotly_dark",
        yaxis=dict(range=[0, 100], title="Weight (%)"),
        title="Current RL Feature Weights",
        height=280,
        margin=dict(l=40, r=30, t=50, b=40),
    )
    st.plotly_chart(fig_weights, use_container_width=True)

    st.divider()

    # ── Accuracy Chart ──
    acc_log = st.session_state.accuracy_log
    has_data = any(len(v) > 0 for v in acc_log.values())

    if not has_data:
        st.info("📈 Accuracy history will appear after the first **🔄 Refresh Data** cycle.")
    else:
        st.markdown("### 📈 Prediction Accuracy Trends")
        st.caption(
            "Each refresh logs a simulated accuracy score. "
            "In production, accuracy is computed by comparing predictions against actual intraday outcomes."
        )

        # Timeframe selector
        tf_options = [tf for tf, data in acc_log.items() if len(data) > 0]
        selected_tfs = st.multiselect(
            "Show timeframes", tf_options,
            default=tf_options,
        )

        filtered_log = {k: v for k, v in acc_log.items() if k in selected_tfs}
        fig_acc = build_accuracy_chart(filtered_log)
        st.plotly_chart(fig_acc, use_container_width=True)

        # Stats table
        st.markdown("### 📋 Accuracy Statistics")
        stats_rows = []
        for tf, data in acc_log.items():
            if data:
                accs = [d["accuracy"] * 100 for d in data]
                stats_rows.append({
                    "Timeframe": tf.capitalize(),
                    "Data Points": len(accs),
                    "Latest (%)": f"{accs[-1]:.2f}",
                    "Average (%)": f"{np.mean(accs):.2f}",
                    "Best (%)": f"{max(accs):.2f}",
                    "Worst (%)": f"{min(accs):.2f}",
                })
        st.dataframe(pd.DataFrame(stats_rows), use_container_width=True, hide_index=True)

    st.divider()

    # ── RL Explanation Panel ──
    with st.expander("📖 How the RL Weight Adjustment Works"):
        st.markdown("""
        ### Simulated Reinforcement Learning Loop

        This application implements a **simplified policy-gradient-inspired** RL feedback cycle:

        | Step | Action |
        |------|--------|
        | **1. Predict** | Generate UP/DOWN predictions using current weights |
        | **2. Observe** | In production: compare prediction vs actual intraday close |
        | **3. Reward** | Positive accuracy trend → +reward; negative → -reward |
        | **4. Update** | Reward reinforces the dominant weight if accuracy improved |
        | **5. Normalise** | `tech_weight + news_weight = 1.0` always |

        #### Weight Update Formula
        ```
        If accuracy improved:
          delta = +0.02 (reinforce dominant feature)
        Else:
          delta = -0.02 (reduce dominant feature weight)

        new_weight = clip(old_weight + delta, min=0.30, max=0.80)
        news_weight = 1.0 - tech_weight
        ```

        #### Production Extension
        Replace `generate_accuracy_point()` with real outcome logging:
        1. At market open: record prediction direction
        2. At market close: compare against `(close - open) / open`
        3. Compute accuracy, update accuracy log, trigger RL step
        """)

    # ── Pipeline Architecture Diagram ──
    with st.expander("🏗️ Prediction Pipeline Architecture"):
        st.markdown("""
        ```
        ┌─────────────────────────────────────────────────────────────────┐
        │                    PREDICTION PIPELINE                          │
        ├───────────────────────────┬─────────────────────────────────────┤
        │   TECHNICAL ARM           │   NLP SENTIMENT ARM                 │
        │                           │                                     │
        │  yfinance.download()      │  yfinance Ticker.news               │
        │       ↓                   │       ↓                             │
        │  RSI (14-period)          │  Headline extraction                │
        │  VWAP (rolling)           │       ↓                             │
        │       ↓                   │  ProsusAI/FinBERT                   │
        │  RSI signal  [-1,0,+1]    │  → positive / negative / neutral    │
        │  VWAP signal [-1,0,+1]    │  net_sentiment = pos - neg          │
        │       ↓                   │       ↓                             │
        │  tech_signal = avg        │  sentiment_signal ∈ [-1, +1]        │
        ├───────────────────────────┴─────────────────────────────────────┤
        │                   COMBINER (RL-weighted)                        │
        │                                                                 │
        │  combined = tech_weight × tech_signal                          │
        │           + news_weight × sentiment_signal                     │
        │                                                                 │
        │  confidence = clip(|combined| × 40 + 50, 50%, 95%)            │
        │  prediction = "UP" if combined > 0 else "DOWN"                 │
        └─────────────────────────────────────────────────────────────────┘
        ```
        """)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "⚠️ **Disclaimer:** This application is for educational and research purposes only. "
    "Predictions do NOT constitute financial advice. Past prediction accuracy does not "
    "guarantee future performance. Always consult a SEBI-registered financial advisor "
    "before making investment decisions.  |  Built with ❤️ using Streamlit + FinBERT + yfinance"
)
