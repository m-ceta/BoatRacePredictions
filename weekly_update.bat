@echo off
setlocal

cd /d "%~dp0"

echo [1/2] Backfilling rowdata...
call boatrace-backfill-rowdata --rowdata rowdata
if errorlevel 1 goto :error

echo [2/2] Rebuilding processed dataset...
call boatrace-build --rowdata rowdata --output data/processed
if errorlevel 1 goto :error

echo Weekly update completed successfully.
exit /b 0

:error
echo Weekly update failed.
exit /b 1
