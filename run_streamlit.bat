@echo off
REM Launcher to start the Streamlit demo on Windows
REM Usage: run_streamlit.bat [port] [server_url] [api_key]

SETLOCAL
SET REPO_DIR=%~dp0
SET PORT=8501
SET SERVERURL=http://127.0.0.1:8000/predict
SET APIKEY=

IF NOT "%1"=="" SET PORT=%~1
IF NOT "%2"=="" SET SERVERURL=%~2
IF NOT "%3"=="" SET APIKEY=%~3

echo Starting Streamlit demo on port %PORT% (server: %SERVERURL%)
SET PYTHONPATH=%REPO_DIR%
IF NOT "%APIKEY%"=="" (
  SET DEMO_API_KEY=%APIKEY%
)
SET DEMO_SERVER_URL=%SERVERURL%
py -3 -m streamlit run "%REPO_DIR%webapp\streamlit_demo.py" --server.port %PORT%

ENDLOCAL
