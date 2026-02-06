@echo off
echo Installing BTC Monitor auto-start...

REM Get script directory
set SCRIPT_DIR=%~dp0

REM Create scheduled task
schtasks /create /tn "BTC Bottom Monitor" /tr "pythonw.exe \"%SCRIPT_DIR%btc_monitor.py\"" /sc onlogon /rl highest /f

echo.
echo Auto-start installed!
echo Monitor will start automatically on next login.
echo.
pause
