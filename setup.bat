@echo off
echo Bitcoin Bottom Detector - Setup
echo ================================

REM Install Python packages
echo.
echo Installing required packages...
pip install -r requirements.txt

REM Create shortcut
echo.
echo Creating desktop shortcut...
powershell "$ws = New-Object -ComObject WScript.Shell; $s = $ws.CreateShortcut('%USERPROFILE%\Desktop\BTC Monitor.lnk'); $s.TargetPath = 'pythonw.exe'; $s.Arguments = '%CD%\btc_monitor.py'; $s.WorkingDirectory = '%CD%'; $s.Save()"

echo.
echo Setup complete!
echo.
echo Next steps:
echo 1. Get Discord webhook URL from Server Settings - Integrations - Webhooks
echo 2. Set environment variable: DISCORD_WEBHOOK_URL
echo 3. Double-click "BTC Monitor" on desktop to start
echo.
pause
