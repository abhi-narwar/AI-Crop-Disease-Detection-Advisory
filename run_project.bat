@echo off
TITLE AI-Enabled Crop Disease Detection and Advisory System
COLOR 0A

echo ======================================================================
echo    AI-ENABLED CROP DISEASE DETECTION AND ADVISORY SYSTEM
echo    B.Tech CSE (AI/ML) Major Project
echo ======================================================================
echo.

cd /d "%~dp0"

echo [1/3] Checking Project Environment...
if exist "venv\Scripts\python.exe" (
    set "PYTHON_EXE=venv\Scripts\python.exe"
) else (
    set "PYTHON_EXE=D:\Anaconda\envs\myenv\python.exe"
)

echo [2/3] Using Python Interpreter: %PYTHON_EXE%
echo [3/3] Starting FastAPI Server on http://127.0.0.1:8000 ...
echo.
echo ======================================================================
echo    WEB APPLICATION URL: http://127.0.0.1:8000
echo    SWAGGER API DOCS:    http://127.0.0.1:8000/docs
echo ======================================================================
echo.
echo Press Ctrl+C in this terminal window to stop the application.
echo.

start "" "http://127.0.0.1:8000"
"%PYTHON_EXE%" -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload

pause
