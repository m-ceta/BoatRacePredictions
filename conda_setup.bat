@echo off
setlocal

cd /d "%~dp0"

set "ENV_NAME=boatrace-predictions"

where conda >nul 2>nul
if errorlevel 1 (
    echo Conda was not found on PATH.
    echo Open an Anaconda Prompt or initialize conda for this shell first.
    exit /b 1
)

echo Checking existing conda environments...
conda env list | findstr /r /c:"^[* ]*%ENV_NAME% " >nul
if errorlevel 1 (
    echo Creating conda environment "%ENV_NAME%" from environment.yml...
    call conda env create -f environment.yml
    if errorlevel 1 goto :error
) else (
    echo Updating conda environment "%ENV_NAME%" from environment.yml...
    call conda env update -n %ENV_NAME% -f environment.yml --prune
    if errorlevel 1 goto :error
)

echo.
echo Setup completed.
echo Activate with:
echo   conda activate %ENV_NAME%
exit /b 0

:error
echo Conda setup failed.
exit /b 1
