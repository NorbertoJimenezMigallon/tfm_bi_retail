@echo off
REM Lanzador para Windows del Dashboard BI Retail TFM
cd /d %~dp0\..
python -m streamlit run app\dashboard.py
pause
