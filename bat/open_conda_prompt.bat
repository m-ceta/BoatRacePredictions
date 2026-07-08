@echo off
setlocal

call "%~dp0_common.bat"
call "%~dp0_common.bat" :activate_conda
if errorlevel 1 exit /b %errorlevel%

title BoatRacePredictions - %ENV_NAME%
echo Activated conda environment "%ENV_NAME%".
echo Project root: %CD%
echo boatrace-* commands are available in this prompt.
cmd /k "cd /d ""%CD%"""
