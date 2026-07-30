@echo off
setlocal

call "%~dp0_common.bat"

if not defined PYTHON_VERSION set "PYTHON_VERSION=3.11"
if not defined INSTALL_PROJECT set "INSTALL_PROJECT=1"
if not defined INSTALL_NN set "INSTALL_NN=1"

call "%~dp0_common.bat" :find_conda
if errorlevel 1 exit /b %errorlevel%

call "%CONDA_BAT%" env list | findstr /R /C:"^%ENV_NAME%[ ]" >nul
if errorlevel 1 (
    echo Creating conda environment "%ENV_NAME%" with Python %PYTHON_VERSION%.
    call "%CONDA_BAT%" create -y -n "%ENV_NAME%" "python=%PYTHON_VERSION%"
    if errorlevel 1 exit /b %errorlevel%
) else (
    echo Conda environment already exists: %ENV_NAME%
)

call "%CONDA_BAT%" activate "%ENV_NAME%"
if errorlevel 1 exit /b %errorlevel%

python -m pip install --upgrade pip
if errorlevel 1 exit /b %errorlevel%

python -m pip install -r requirements.txt
if errorlevel 1 exit /b %errorlevel%

if "%INSTALL_PROJECT%"=="1" (
    if "%INSTALL_NN%"=="1" (
        python -m pip install -e ".[nn]"
    ) else (
        python -m pip install -e .
    )
    if errorlevel 1 exit /b %errorlevel%
)

echo Windows environment setup completed.
echo Activate with:
echo   conda activate %ENV_NAME%
exit /b 0
