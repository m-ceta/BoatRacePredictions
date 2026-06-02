@echo off
setlocal

cd /d "%~dp0.."

set "ENV_NAME=boatrace-predictions"
set "CONDA_BAT="

for /f "delims=" %%I in ('where conda.bat 2^>nul') do (
    set "CONDA_BAT=%%I"
    goto :found_conda
)

for /f "delims=" %%I in ('where conda 2^>nul') do (
    set "CONDA_BAT=%%~dpI..\condabin\conda.bat"
    goto :found_conda
)

echo Conda was not found on PATH.
exit /b 1

:found_conda
if not exist "%CONDA_BAT%" (
    echo conda.bat was not found.
    echo Expected: %CONDA_BAT%
    exit /b 1
)

call "%CONDA_BAT%" activate %ENV_NAME%
if errorlevel 1 (
    echo Failed to activate conda environment "%ENV_NAME%".
    exit /b 1
)

echo [1/3] boatrace-backfill-rowdata --rowdata rowdata
call boatrace-backfill-rowdata --rowdata rowdata
if errorlevel 1 exit /b %errorlevel%

echo [2/3] boatrace-build --rowdata rowdata --output data/processed
call boatrace-build --rowdata rowdata --output data/processed
if errorlevel 1 exit /b %errorlevel%

echo [3/3] boatrace-package-upload
call boatrace-package-upload
if errorlevel 1 exit /b %errorlevel%

echo data_build.bat completed successfully.
exit /b 0
