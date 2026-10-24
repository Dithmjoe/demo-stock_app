# 📈 Intraday Stock Prediction System (Web & Android Mobile)

A complete stock prediction platform that combines **Technical Analysis** (RSI + VWAP) with **Financial News Sentiment (FinBERT NLP)** to forecast intraday UP/DOWN directional trends for Indian (NSE) stocks, enhanced by a **Reinforcement Learning** adaptive weight loop.

Available as:
1. **📱 Android Mobile Application** (Built with Kivy, touch-friendly, dark terminal UI)
2. **💻 Web Application** (Streamlit dashboard with interactive Plotly charts)

---

## 📱 Android Mobile App (Kivy)

The mobile application is built in Python using **Kivy**, featuring a sleek dark-themed mobile terminal UI designed for smartphones.

### 🌟 Mobile Features
- **Nifty-50 Live Universe**: Instant stock cards with price, % change, prediction badges (🟢 UP / 🔴 DOWN), and confidence ratings.
- **Real-Time Search**: Filter stocks instantly by company name, ticker, or sector.
- **Stock Detail Modal**: Tap any stock to view detailed technical indicators (RSI 14, VWAP), live news headlines, and combined score formulas.
- **Watchlist & Bookmarks**: Add/remove any custom NSE ticker (e.g. `TATAMOTORS.NS`, `ZOMATO.NS`) with real-time tracking.
- **Reinforcement Learning & AI Settings**:
  - Live RL weight adjustments (`tech_weight` vs `news_weight`).
  - **Option B FinBERT Integration**: Uses Hugging Face's **Free Serverless Inference API** for deep learning sentiment, with an automatic offline rule-based fallback if offline.
  - Indian Market Hours status indicator (NSE Open 9:15 AM - 3:30 PM IST).

### 🚀 Running the Mobile App on Desktop (Windows)
```bash
# Run the mobile app directly on your screen (mobile emulator window)
run_mobile.bat

# OR via python command
python main.py
```

### 📦 Building the Android APK (`.apk`)

#### Method 1: Automatic Cloud Build via GitHub Actions (Recommended)
You do **not** need Linux or Android NDK installed on your Windows PC!
1. Push this repository to your GitHub account (`git push origin main`).
2. Go to your repository on GitHub and click the **Actions** tab.
3. The **"Build Android APK"** workflow will automatically compile the APK using Ubuntu runners.
4. Once completed, download the `StockPredictor-Android-APK` artifact and install it directly on your Android phone!

#### Method 2: Local Build with Buildozer (Linux / WSL2)
If you have WSL2 or Ubuntu Linux:
```bash
pip install buildozer cython
buildozer -v android debug
```
The compiled APK will be created inside the `bin/` directory.

---

## 💻 Web Application (Streamlit)

### Quick Start
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the web app
streamlit run app.py

# OR use the batch file (Windows)
run_app.bat
```
The web dashboard opens at **http://localhost:8501**.

---

## 🧠 Prediction Architecture

```
OHLCV (yfinance) ────────► RSI (14) + VWAP ──────────► Tech Signal [-1, +1]
                                                              │
                                                       RL-Weighted Combiner
                                                              │
News (Google RSS / Yahoo) ─► FinBERT API (or Rule NLP) ─► Sentiment [-1, +1]
                                                              │
                                                       UP / DOWN + Confidence %
```

---

## ⚙️ Free APIs & Services Used

| Component | Source | Cost | API Key Required? |
| :--- | :--- | :--- | :--- |
| **Stock Data** | Yahoo Finance (`yfinance`) | **$0 (Free)** | No |
| **News Headlines** | Google News RSS + Yahoo News | **$0 (Free)** | No |
| **Technical Math** | Pure Python / NumPy | **$0 (Free)** | No (Runs on-device) |
| **FinBERT NLP** | Hugging Face Serverless API | **$0 (Free)** | Free user token (`hf_...`) |
| **Offline Sentiment** | Built-in Financial Keyword Lexicon | **$0 (Free)** | No |

---

## ⚠️ Disclaimer

This application is for **educational and research purposes only**.  
Predictions do **not** constitute financial advice. Always consult a SEBI-registered financial advisor before investing.
