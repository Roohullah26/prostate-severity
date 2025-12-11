@echo off
REM Start FastAPI Server from deploy_clean folder

echo.
echo ====================================================
echo   Starting Prostate Severity Analyzer - API Server
echo ====================================================
echo.

cd /d "d:\prostate project\prostate-severity\deploy_clean"

REM Run with venv Python
"D:\prostate project\prostate-severity\venv\Scripts\python.exe" -m uvicorn webapp.fastapi_server:app --host 127.0.0.1 --port 8000 --reload

REM If error, keep window open
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Failed to start server
    pause
)
