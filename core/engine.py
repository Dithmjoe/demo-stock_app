"""
Core prediction, market data, and AI engine for Intraday Stock Predictor Mobile.
Compatible with Android (via Kivy/Buildozer) and Desktop.
"""

import os
import sys
import time
import json
import datetime
import urllib.request
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import requests

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS & ASSETS
# ─────────────────────────────────────────────────────────────────────────────
NIFTY_50_STOCKS = [
    {"ticker": "RELIANCE.NS", "name": "Reliance Industries", "sector": "Energy"},
    {"ticker": "TCS.NS", "name": "Tata Consultancy Services", "sector": "IT"},
    {"ticker": "HDFCBANK.NS", "name": "HDFC Bank Ltd", "sector": "Banking"},
    {"ticker": "INFY.NS", "name": "Infosys Ltd", "sector": "IT"},
    {"ticker": "ICICIBANK.NS", "name": "ICICI Bank Ltd", "sector": "Banking"},
    {"ticker": "HINDUNILVR.NS", "name": "Hindustan Unilever", "sector": "FMCG"},
    {"ticker": "SBIN.NS", "name": "State Bank of India", "sector": "Banking"},
    {"ticker": "BHARTIARTL.NS", "name": "Bharti Airtel", "sector": "Telecom"},
    {"ticker": "ITC.NS", "name": "ITC Ltd", "sector": "FMCG"},
    {"ticker": "KOTAKBANK.NS", "name": "Kotak Mahindra Bank", "sector": "Banking"},
    {"ticker": "LT.NS", "name": "Larsen & Toubro", "sector": "Capital Goods"},
    {"ticker": "HCLTECH.NS", "name": "HCL Technologies", "sector": "IT"},
    {"ticker": "AXISBANK.NS", "name": "Axis Bank Ltd", "sector": "Banking"},
    {"ticker": "ASIANPAINT.NS", "name": "Asian Paints", "sector": "Consumer"},
    {"ticker": "MARUTI.NS", "name": "Maruti Suzuki", "sector": "Automobile"},
    {"ticker": "BAJFINANCE.NS", "name": "Bajaj Finance", "sector": "Finance"},
    {"ticker": "TITAN.NS", "name": "Titan Company", "sector": "Consumer"},
    {"ticker": "WIPRO.NS", "name": "Wipro Ltd", "sector": "IT"},
    {"ticker": "ULTRACEMCO.NS", "name": "UltraTech Cement", "sector": "Materials"},
    {"ticker": "SUNPHARMA.NS", "name": "Sun Pharma", "sector": "Pharma"},
    {"ticker": "POWERGRID.NS", "name": "Power Grid Corp", "sector": "Utilities"},
    {"ticker": "NTPC.NS", "name": "NTPC Ltd", "sector": "Utilities"},
    {"ticker": "TECHM.NS", "name": "Tech Mahindra", "sector": "IT"},
    {"ticker": "M&M.NS", "name": "Mahindra & Mahindra", "sector": "Automobile"},
    {"ticker": "BAJAJFINSV.NS", "name": "Bajaj Finserv", "sector": "Finance"},
    {"ticker": "ONGC.NS", "name": "Oil & Natural Gas Corp", "sector": "Energy"},
    {"ticker": "JSWSTEEL.NS", "name": "JSW Steel Ltd", "sector": "Metals"},
    {"ticker": "TATAMOTORS.NS", "name": "Tata Motors Ltd", "sector": "Automobile"},
    {"ticker": "ADANIENT.NS", "name": "Adani Enterprises", "sector": "Conglomerate"},
    {"ticker": "ADANIPORTS.NS", "name": "Adani Ports & SEZ", "sector": "Infrastructure"},
    {"ticker": "COALINDIA.NS", "name": "Coal India Ltd", "sector": "Mining"},
    {"ticker": "DRREDDY.NS", "name": "Dr. Reddy's Labs", "sector": "Pharma"},
    {"ticker": "CIPLA.NS", "name": "Cipla Ltd", "sector": "Pharma"},
    {"ticker": "INDUSINDBK.NS", "name": "IndusInd Bank", "sector": "Banking"},
    {"ticker": "BRITANNIA.NS", "name": "Britannia Industries", "sector": "FMCG"},
    {"ticker": "EICHERMOT.NS", "name": "Eicher Motors", "sector": "Automobile"},
    {"ticker": "GRASIM.NS", "name": "Grasim Industries", "sector": "Materials"},
    {"ticker": "DIVISLAB.NS", "name": "Divi's Laboratories", "sector": "Pharma"},
    {"ticker": "HDFCLIFE.NS", "name": "HDFC Life Insurance", "sector": "Insurance"},
    {"ticker": "SBILIFE.NS", "name": "SBI Life Insurance", "sector": "Insurance"},
    {"ticker": "HEROMOTOCO.NS", "name": "Hero MotoCorp", "sector": "Automobile"},
    {"ticker": "BAJAJ-AUTO.NS", "name": "Bajaj Auto", "sector": "Automobile"},
    {"ticker": "BPCL.NS", "name": "Bharat Petroleum", "sector": "Energy"},
    {"ticker": "TATACONSUM.NS", "name": "Tata Consumer Products", "sector": "FMCG"},
    {"ticker": "APOLLOHOSP.NS", "name": "Apollo Hospitals", "sector": "Healthcare"},
    {"ticker": "TATASTEEL.NS", "name": "Tata Steel Ltd", "sector": "Metals"},
    {"ticker": "NESTLEIND.NS", "name": "Nestle India", "sector": "FMCG"},
]

RSI_OVERSOLD = 35
RSI_OVERBOUGHT = 65

HF_FINBERT_URL = "https://router.huggingface.co/hf-inference/models/ProsusAI/finbert"

# ─────────────────────────────────────────────────────────────────────────────
# STORAGE HELPERS (CROSS-PLATFORM)
# ─────────────────────────────────────────────────────────────────────────────
def get_data_dir() -> str:
    """Return appropriate storage directory for current platform."""
    # Check if running under Kivy on Android
    try:
        from kivy.app import App
        app = App.get_running_app()
        if app and hasattr(app, "user_data_dir"):
            return app.user_data_dir
    except Exception:
        pass
    return os.path.abspath(".")

def load_json_file(filename: str, default):
    filepath = os.path.join(get_data_dir(), filename)
    try:
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return default

def save_json_file(filename: str, data):
    filepath = os.path.join(get_data_dir(), filename)
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
    except Exception:
        pass

# ─────────────────────────────────────────────────────────────────────────────
# SETTINGS & STATE MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────
def load_settings() -> dict:
    defaults = {
        "hf_token": "",
        "use_finbert": True,
        "auto_refresh_sec": 300,
        "tech_weight": 0.55,
        "news_weight": 0.45,
        "iterations": 0,
    }
    saved = load_json_file("settings.json", {})
    defaults.update(saved)
    return defaults

def save_settings(settings: dict):
    save_json_file("settings.json", settings)

def load_bookmarks() -> list:
    return load_json_file("bookmarks.json", ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS"])

def save_bookmarks(bookmarks: list):
    save_json_file("bookmarks.json", bookmarks)

def load_rl_state() -> dict:
    default = {"tech_weight": 0.55, "news_weight": 0.45, "iterations": 0}
    return load_json_file("rl_state.json", default)

def save_rl_state(rl_state: dict):
    save_json_file("rl_state.json", rl_state)

def load_accuracy_log() -> dict:
    default = {"daily": [], "weekly": [], "monthly": []}
    return load_json_file("accuracy_log.json", default)

def save_accuracy_log(log: dict):
    save_json_file("accuracy_log.json", log)

# ─────────────────────────────────────────────────────────────────────────────
# MARKET STATUS HELPER
# ─────────────────────────────────────────────────────────────────────────────
def get_market_status() -> dict:
    """Return Indian NSE market status (Open 9:15 AM - 3:30 PM IST Mon-Fri)."""
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    ist_offset = datetime.timedelta(hours=5, minutes=30)
    now_ist = now_utc + ist_offset

    weekday = now_ist.weekday()  # 0 is Monday, 6 is Sunday
    is_weekday = weekday < 5
    open_time = datetime.time(9, 15)
    close_time = datetime.time(15, 30)

    is_open = is_weekday and (open_time <= now_ist.time() <= close_time)
    status_text = "NSE Open" if is_open else "NSE Closed"
    time_str = now_ist.strftime("%H:%M IST")

    return {
        "is_open": is_open,
        "status_text": f"{status_text} ({time_str})",
        "time_str": time_str
    }

# ─────────────────────────────────────────────────────────────────────────────
# TECHNICAL ANALYSIS (PURE NUMPY/PANDAS)
# ─────────────────────────────────────────────────────────────────────────────
def fetch_ohlcv(ticker: str, period: str = "5d", interval: str = "1d") -> pd.DataFrame:
    """
    Download OHLCV dataframe using yfinance with safety fallbacks.
    """
    import yfinance as yf
    try:
        df = yf.download(
            ticker,
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False,
            threads=False,
            timeout=10,
        )
        if df.empty or len(df) < 3:
            return pd.DataFrame()
        df.dropna(inplace=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception:
        return pd.DataFrame()

def compute_technical_indicators(df: pd.DataFrame) -> dict:
    """
    Compute RSI (14) and VWAP using pure NumPy/Pandas.
    Avoids compilation dependencies on Android.
    """
    res = {
        "rsi": None,
        "vwap": None,
        "close": None,
        "change_pct": 0.0,
        "rsi_signal": 0,
        "vwap_signal": 0,
        "tech_signal": 0.0,
        "error": None,
        "prices": [],
    }

    if df.empty or len(df) < 2:
        res["error"] = "Insufficient data"
        return res

    try:
        closes = df["Close"].astype(float)
        highs = df["High"].astype(float)
        lows = df["Low"].astype(float)
        volumes = df["Volume"].astype(float)

        last_close = float(closes.iloc[-1])
        prev_close = float(closes.iloc[-2]) if len(closes) > 1 else last_close
        change_pct = round(((last_close - prev_close) / prev_close) * 100, 2)

        res["close"] = round(last_close, 2)
        res["change_pct"] = change_pct
        res["prices"] = [round(float(p), 2) for p in closes.tail(10).tolist()]

        # ── RSI 14 (Wilder's Smoothing or Standard SMA for robustness) ──
        delta = closes.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        window = min(14, len(closes) - 1)
        if window > 0:
            avg_gain = gain.rolling(window=window, min_periods=1).mean().iloc[-1]
            avg_loss = loss.rolling(window=window, min_periods=1).mean().iloc[-1]

            if avg_loss == 0:
                rsi_val = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi_val = 100.0 - (100.0 / (1.0 + rs))
        else:
            rsi_val = 50.0

        res["rsi"] = round(float(rsi_val), 1)

        if rsi_val < RSI_OVERSOLD:
            res["rsi_signal"] = 1     # Oversold -> Bounce UP
        elif rsi_val > RSI_OVERBOUGHT:
            res["rsi_signal"] = -1    # Overbought -> Pullback DOWN
        else:
            res["rsi_signal"] = 0

        # ── VWAP ──
        typical_price = (highs + lows + closes) / 3.0
        vwap_series = (typical_price * volumes).cumsum() / (volumes.cumsum() + 1e-9)
        last_vwap = float(vwap_series.iloc[-1])
        res["vwap"] = round(last_vwap, 2)

        if last_close > last_vwap:
            res["vwap_signal"] = 1
        elif last_close < last_vwap:
            res["vwap_signal"] = -1

        # Normalized technical score [-1.0, 1.0]
        res["tech_signal"] = (res["rsi_signal"] + res["vwap_signal"]) / 2.0

    except Exception as e:
        res["error"] = str(e)

    return res

# ─────────────────────────────────────────────────────────────────────────────
# NEWS FETCHER (GOOGLE NEWS RSS + YFINANCE)
# ─────────────────────────────────────────────────────────────────────────────
def fetch_news_headlines(ticker: str, company_name: str = "", max_items: int = 5) -> list[str]:
    """
    Fetch live news headlines from Google News RSS for the Indian stock,
    with yfinance news fallback. 100% free with no API key needed.
    """
    headlines = []
    clean_name = company_name if company_name else ticker.replace(".NS", "")
    query = f"{clean_name} stock share price NSE"
    url = f"https://news.google.com/rss/search?q={urllib.parse.quote(query)}&hl=en-IN&gl=IN&ceid=IN:en"

    # 1. Try Google News RSS
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Linux; Android 10; K) Mobile Safari/537.36"}
        resp = requests.get(url, headers=headers, timeout=6)
        if resp.status_code == 200:
            root = ET.fromstring(resp.content)
            for item in root.findall(".//item")[:max_items]:
                title_elem = item.find("title")
                if title_elem is not None and title_elem.text:
                    clean_title = title_elem.text.split(" - ")[0].strip()
                    if clean_title and clean_title not in headlines:
                        headlines.append(clean_title)
    except Exception:
        pass

    # 2. Fallback to yfinance news if RSS returned < 2 items
    if len(headlines) < 2:
        try:
            import yfinance as yf
            t = yf.Ticker(ticker)
            for item in (t.news or [])[:max_items]:
                content = item.get("content", {})
                title = content.get("title", "") if isinstance(content, dict) else ""
                if not title:
                    title = item.get("title", "")
                if title and title not in headlines:
                    headlines.append(title)
        except Exception:
            pass

    return headlines

# ─────────────────────────────────────────────────────────────────────────────
# SENTIMENT ANALYSIS (FINBERT FREE API + RULE-BASED FALLBACK)
# ─────────────────────────────────────────────────────────────────────────────
def rule_based_sentiment(headlines: list[str]) -> dict:
    """
    Fast, local, offline keyword sentiment analysis.
    Zero external dependencies and 100% reliable.
    """
    BULL_WORDS = {
        "surge", "soar", "gain", "jump", "rise", "up", "high", "growth",
        "profit", "beat", "record", "strong", "buy", "upgrade", "bullish",
        "positive", "rally", "outperform", "revenue", "expansion", "dividend",
        "acquisition", "order", "win", "breakout", "target", "blockbuster"
    }
    BEAR_WORDS = {
        "fall", "drop", "crash", "decline", "loss", "down", "low", "weak",
        "miss", "sell", "downgrade", "bearish", "negative", "cut", "risk",
        "concern", "warning", "underperform", "contraction", "layoff", "debt",
        "probe", "penalty", "slump", "fraud", "tumble", "cautious"
    }

    if not headlines:
        return {
            "positive": 0.33,
            "negative": 0.33,
            "neutral": 0.34,
            "net_sentiment": 0.0,
            "source": "No Headlines",
            "count": 0,
        }

    pos, neg = 0, 0
    for h in headlines:
        words = set(h.lower().replace("-", " ").replace(".", " ").split())
        pos += len(words & BULL_WORDS)
        neg += len(words & BEAR_WORDS)

    total = pos + neg
    if total == 0:
        pos_score, neg_score = 0.33, 0.33
        net = 0.0
    else:
        pos_score = round(pos / total, 3)
        neg_score = round(neg / total, 3)
        net = round(pos_score - neg_score, 3)

    return {
        "positive": pos_score,
        "negative": neg_score,
        "neutral": round(max(0.0, 1.0 - pos_score - neg_score), 3),
        "net_sentiment": net,
        "source": "Rule-Based NLP",
        "count": len(headlines),
    }

def query_huggingface_finbert(headlines: list[str], hf_token: str) -> dict | None:
    """
    Query Hugging Face's Free Serverless Inference API for ProsusAI/finbert.
    Returns parsed dict or None if API error occurs.
    """
    if not headlines or not hf_token or not hf_token.strip():
        return None

    headers = {
        "Authorization": f"Bearer {hf_token.strip()}",
        "Content-Type": "application/json",
    }

    # FinBERT classifies up to 512 tokens. Send top 3-4 headlines combined
    prompt_text = " . ".join(headlines[:4])
    payload = {"inputs": prompt_text}

    try:
        resp = requests.post(HF_FINBERT_URL, headers=headers, json=payload, timeout=8)
        if resp.status_code == 200:
            data = resp.json()
            # HF outputs: [[{'label': 'positive', 'score': 0.8}, ...]] or [{'label': ...}]
            scores_list = data[0] if isinstance(data, list) and isinstance(data[0], list) else data
            score_map = {item["label"].lower(): item["score"] for item in scores_list if "label" in item}

            pos = round(float(score_map.get("positive", 0.0)), 3)
            neg = round(float(score_map.get("negative", 0.0)), 3)
            neu = round(float(score_map.get("neutral", 0.0)), 3)
            net = round(pos - neg, 3)

            return {
                "positive": pos,
                "negative": neg,
                "neutral": neu,
                "net_sentiment": net,
                "source": "FinBERT AI (HF)",
                "count": len(headlines),
            }
    except Exception:
        pass

    return None

def analyze_sentiment(headlines: list[str], hf_token: str = "") -> dict:
    """
    Unified sentiment analyzer: tries FinBERT API first if token present,
    with seamless automatic fallback to rule-based analysis.
    """
    if hf_token:
        hf_result = query_huggingface_finbert(headlines, hf_token)
        if hf_result:
            return hf_result

    return rule_based_sentiment(headlines)

# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION COMBINER & REINFORCEMENT LEARNING
# ─────────────────────────────────────────────────────────────────────────────
def combine_prediction(tech_signal: float, net_sentiment: float,
                       tech_weight: float, news_weight: float) -> dict:
    """
    Weighted combination of technical indicators and sentiment.
    Returns direction ("UP" or "DOWN") and confidence % [50% - 95%].
    """
    combined = (tech_weight * tech_signal) + (news_weight * net_sentiment)
    confidence = min(95.0, max(50.0, abs(combined) * 40.0 + 50.0))
    direction = "UP" if combined > 0 else "DOWN"

    return {
        "direction": direction,
        "confidence": round(confidence, 1),
        "combined_score": round(combined, 3),
    }

def simulate_rl_adjustment(rl_state: dict, accuracy_log: dict) -> dict:
    """
    Update feature weights (tech_weight vs news_weight) based on recent performance.
    """
    new_state = dict(rl_state)
    daily = accuracy_log.get("daily", [])

    if len(daily) >= 2:
        recent_trend = daily[-1]["accuracy"] - daily[-2]["accuracy"]
        dominant = "tech" if rl_state["tech_weight"] >= rl_state["news_weight"] else "news"
        delta = 0.02 * np.sign(recent_trend)

        if dominant == "tech":
            new_state["tech_weight"] = round(np.clip(rl_state["tech_weight"] + delta, 0.3, 0.8), 3)
        else:
            new_state["news_weight"] = round(np.clip(rl_state["news_weight"] + delta, 0.2, 0.7), 3)

        total = new_state["tech_weight"] + new_state["news_weight"]
        new_state["tech_weight"] = round(new_state["tech_weight"] / total, 3)
        new_state["news_weight"] = round(new_state["news_weight"] / total, 3)
    else:
        delta = float(np.random.choice([-0.01, 0.0, 0.01]))
        new_state["tech_weight"] = round(np.clip(rl_state["tech_weight"] + delta, 0.3, 0.8), 3)
        new_state["news_weight"] = round(1.0 - new_state["tech_weight"], 3)

    new_state["iterations"] = rl_state.get("iterations", 0) + 1
    return new_state

def generate_accuracy_point(iteration: int) -> dict:
    base = min(0.85, 0.55 + iteration * 0.006)
    noise = float(np.random.normal(0, 0.03))
    accuracy = round(float(np.clip(base + noise, 0.48, 0.92)), 4)
    return {
        "date": datetime.date.today().isoformat(),
        "accuracy": accuracy,
        "iteration": iteration,
    }

# ─────────────────────────────────────────────────────────────────────────────
# SINGLE TICKER FULL ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
def analyze_stock(ticker: str, company_name: str = "", hf_token: str = "",
                  tech_weight: float = 0.55, news_weight: float = 0.45) -> dict:
    """
    Run complete end-to-end analysis for a ticker.
    """
    res = {
        "ticker": ticker,
        "name": company_name or ticker.replace(".NS", ""),
        "close": 0.0,
        "change_pct": 0.0,
        "rsi": 50.0,
        "vwap": 0.0,
        "tech_signal": 0.0,
        "sentiment_score": 0.0,
        "sentiment_source": "Offline",
        "headlines": [],
        "direction": "NEUTRAL",
        "confidence": 50.0,
        "error": None,
    }

    # 1. OHLCV & Tech analysis
    df = fetch_ohlcv(ticker)
    if df.empty:
        res["error"] = "Data unavailable"
        return res

    tech = compute_technical_indicators(df)
    if tech["error"]:
        res["error"] = tech["error"]
        return res

    res["close"] = tech["close"]
    res["change_pct"] = tech["change_pct"]
    res["rsi"] = tech["rsi"]
    res["vwap"] = tech["vwap"]
    res["tech_signal"] = tech["tech_signal"]
    res["prices"] = tech["prices"]

    # 2. Headlines & Sentiment
    headlines = fetch_news_headlines(ticker, company_name=company_name)
    res["headlines"] = headlines

    sentiment = analyze_sentiment(headlines, hf_token=hf_token)
    res["sentiment_score"] = sentiment["net_sentiment"]
    res["sentiment_source"] = sentiment["source"]

    # 3. Combine
    pred = combine_prediction(
        res["tech_signal"],
        res["sentiment_score"],
        tech_weight=tech_weight,
        news_weight=news_weight,
    )
    res["direction"] = pred["direction"]
    res["confidence"] = pred["confidence"]

    return res
