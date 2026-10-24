@echo off
echo Starting Intraday Stock Prediction App...
echo.
echo Make sure all requirements are installed:
echo   pip install -r requirements.txt
echo.
streamlit run app.py --server.port 8501 --server.headless false
pause
