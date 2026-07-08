@echo off
setlocal

call "%~dp0_common.bat"
call "%~dp0_common.bat" :activate_conda
if errorlevel 1 exit /b %errorlevel%

echo [1/1] boatrace-webui
call boatrace-webui
if errorlevel 1 exit /b %errorlevel%

echo full_ui.bat completed successfully.
exit /b 0
