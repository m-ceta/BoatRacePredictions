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
echo Open an Anaconda Prompt or initialize conda for this shell first.
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
    echo Run bat\conda_setup.bat first if the environment has not been created yet.
    exit /b 1
)

title BoatRacePredictions - %ENV_NAME%
echo Activated conda environment "%ENV_NAME%".
echo Project root: %CD%
echo boatrace-* commands are available in this prompt.
cmd /k "cd /d %CD%"
