@echo off
setlocal

call "%~dp0_common.bat"

if not defined PYTHON_VERSION set "PYTHON_VERSION=3.11"
if not defined INSTALL_PROJECT set "INSTALL_PROJECT=1"
if not defined INSTALL_NN set "INSTALL_NN=1"
if not defined PYTORCH_DEVICE set "PYTORCH_DEVICE=auto"
if not defined PYTORCH_CUDA_VERSION set "PYTORCH_CUDA_VERSION=cu121"

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

if "%INSTALL_NN%"=="1" (
    call :install_neural_dependencies
    if errorlevel 1 exit /b %errorlevel%
)

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

:install_neural_dependencies
set "RESOLVED_PYTORCH_DEVICE=%PYTORCH_DEVICE%"
if /I "%PYTORCH_DEVICE%"=="cuda" set "RESOLVED_PYTORCH_DEVICE=gpu"
if /I "%PYTORCH_DEVICE%"=="auto" (
    where nvidia-smi >nul 2>nul
    if errorlevel 1 (
        set "RESOLVED_PYTORCH_DEVICE=cpu"
    ) else (
        set "RESOLVED_PYTORCH_DEVICE=gpu"
    )
)

if /I "%RESOLVED_PYTORCH_DEVICE%"=="gpu" (
    set "DEFAULT_PYTORCH_INDEX_URL=https://download.pytorch.org/whl/%PYTORCH_CUDA_VERSION%"
) else if /I "%RESOLVED_PYTORCH_DEVICE%"=="cpu" (
    set "DEFAULT_PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cpu"
) else (
    echo Unsupported PYTORCH_DEVICE: %PYTORCH_DEVICE%. Use cpu, gpu, cuda, or auto.
    exit /b 1
)

if defined PYTORCH_INDEX_URL (
    set "RESOLVED_PYTORCH_INDEX_URL=%PYTORCH_INDEX_URL%"
) else (
    set "RESOLVED_PYTORCH_INDEX_URL=%DEFAULT_PYTORCH_INDEX_URL%"
)

echo Installing neural dependencies: PYTORCH_DEVICE=%RESOLVED_PYTORCH_DEVICE%, index=%RESOLVED_PYTORCH_INDEX_URL%
python -m pip uninstall -y torch torchvision torchaudio >nul 2>nul
python -m pip install --index-url "%RESOLVED_PYTORCH_INDEX_URL%" torch
if errorlevel 1 exit /b %errorlevel%
python -m pip install "pytorch-tabnet>=4.1.0"
if errorlevel 1 exit /b %errorlevel%
exit /b 0
