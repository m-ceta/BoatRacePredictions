@echo off
setlocal

cd /d "%~dp0"

call :resolve_python
if errorlevel 1 goto :error

echo [1/4] Backfilling rowdata...
call "%PYTHON_CMD%" %PYTHON_ARGS% -c "from src.cli import backfill_rowdata_main; backfill_rowdata_main()" --rowdata rowdata
if errorlevel 1 goto :error

echo [2/4] Rebuilding processed dataset...
call "%PYTHON_CMD%" %PYTHON_ARGS% -c "from src.cli import build_dataset_main; build_dataset_main()" --rowdata rowdata --output data/processed
if errorlevel 1 goto :error

echo [3/4] Retraining ranker models...
call "%PYTHON_CMD%" %PYTHON_ARGS% -c "from src.cli import train_main; train_main()" --config configs/train.yaml
if errorlevel 1 goto :error

echo [4/4] Updating trifecta Phase3 models...
call "%PYTHON_CMD%" %PYTHON_ARGS% -c "from src.cli import train_trifecta_v2_main; train_trifecta_v2_main()" --config configs/train.yaml --max-races 1000 --eval-max-races 1000 --eval-rerank-top-n 10 --optimize-rerank
if errorlevel 1 goto :error

echo Monthly update completed successfully.
exit /b 0

:resolve_python
if exist "%~dp0.venv\Scripts\python.exe" (
    set "PYTHON_CMD=%~dp0.venv\Scripts\python.exe"
    set "PYTHON_ARGS="
    exit /b 0
)
if exist "%~dp0venv\Scripts\python.exe" (
    set "PYTHON_CMD=%~dp0venv\Scripts\python.exe"
    set "PYTHON_ARGS="
    exit /b 0
)
where py >nul 2>nul
if not errorlevel 1 (
    set "PYTHON_CMD=py"
    set "PYTHON_ARGS=-3"
    exit /b 0
)
where python >nul 2>nul
if not errorlevel 1 (
    set "PYTHON_CMD=python"
    set "PYTHON_ARGS="
    exit /b 0
)
echo Python runtime was not found. Install Python or create .venv\Scripts\python.exe.
exit /b 1

:error
echo Monthly update failed.
exit /b 1
