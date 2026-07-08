@echo off
setlocal

cd /d "%~dp0.."
start "BoatRacePredictions train_full" /min cmd /c "call bat\train_full.bat > script.log 2>&1"
echo train_full.bat started in background. Log: script.log
exit /b 0
