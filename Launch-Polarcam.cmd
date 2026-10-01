@echo off
setlocal
cd /d "%~dp0"
if not exist "%~dp0.venv\Scripts\python.exe" (
    echo Polarcam's project-local Python environment is missing.
    echo See LAB_SETUP.md in this folder for installation steps.
    pause
    exit /b 1
)
set "PYTHONUNBUFFERED=1"
"%~dp0.venv\Scripts\python.exe" -u -m polarcam --data-dir "%~dp0runs" %*
set "RESULT=%ERRORLEVEL%"
if not "%RESULT%"=="0" (
    echo.
    echo Polarcam exited with code %RESULT%. Keep the error above for troubleshooting.
    pause
)
exit /b %RESULT%