@echo off
title He Thong Giam Sat Dung Do Xe - Dashboard
echo ============================================================
echo   DANG KHOI DONG DASHBOARD GIAM SAT DUNG DO XE THONG MINH
echo ============================================================
echo Dang mo Streamlit Dashboard...
echo.

if exist .venv\Scripts\python.exe (
    .\.venv\Scripts\python.exe -m streamlit run src\dashboard.py
) else (
    python -m streamlit run src\dashboard.py
)

pause
