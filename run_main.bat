@echo off
title He Thong Giam Sat Dung Do Xe - CLI
echo ============================================================
echo   DANG KHOI DONG HE THONG GIAM SAT DUNG DO XE (OPENCV)
echo ============================================================
echo Dang khoi dong YOLO va theo doi xe tren video mau...
echo.

if exist .venv\Scripts\python.exe (
    .\.venv\Scripts\python.exe src\main.py
) else (
    python src\main.py
)

pause
