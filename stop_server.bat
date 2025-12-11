@echo off
REM Stop process(es) listening on a port on Windows (uses netstat & taskkill)
REM Usage: stop_server.bat [port]

SET PORT=8000
IF NOT "%1"=="" SET PORT=%~1

echo Looking for processes bound to port %PORT% ...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":%PORT%"') do (
  echo Found PID %%a, attempting to terminate
  taskkill /PID %%a /F
)

echo Done.
