@echo off
REM Launcher to start the API server on Windows without PowerShell execution policy changes.
REM Usage: run_server.bat [state_path] [in_ch] [host] [port]

SETLOCAL
SET REPO_DIR=%~dp0
REM defaults
SET STATE=%REPO_DIR%models\prototype_toy.pth
SET INCH=3
SET HOST=127.0.0.1
SET PORT=8000

IF NOT "%1"=="" SET STATE=%~1
IF NOT "%2"=="" SET INCH=%~2
IF NOT "%3"=="" SET HOST=%~3
IF NOT "%4"=="" SET PORT=%~4

echo Starting server with:
echo   PYTHONPATH=%REPO_DIR%
echo   state=%STATE%
echo   in_channels=%INCH%
echo   host=%HOST% port=%PORT%

SET PYTHONPATH=%REPO_DIR%
py -3 -m webapp.fastapi_server --state-dict "%STATE%" --in-ch %INCH% --host %HOST% --port %PORT%

ENDLOCAL
