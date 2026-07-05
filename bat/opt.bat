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

echo [1/1] boatrace-train-trifecta-v2 --config configs/train.yaml --max-races 1000 --eval-max-races 3000 --optimize-rerank
call boatrace-train-trifecta-v2 --config configs/train.yaml --max-races 1000 --eval-max-races 3000 --optimize-rerank
if errorlevel 1 exit /b %errorlevel%

echo opt.bat completed successfully.
exit /b 0
