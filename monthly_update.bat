@echo off
setlocal

cd /d "%~dp0"

echo [1/4] Backfilling rowdata...
call boatrace-backfill-rowdata --rowdata rowdata
if errorlevel 1 goto :error

echo [2/4] Rebuilding processed dataset...
call boatrace-build --rowdata rowdata --output data/processed
if errorlevel 1 goto :error

echo [3/4] Retraining ranker models...
call boatrace-train --config configs/train.yaml
if errorlevel 1 goto :error

echo [4/4] Updating trifecta Phase3 models...
call boatrace-train-trifecta-v2 --config configs/train.yaml --max-races 1000 --eval-max-races 1000 --eval-rerank-top-n 10 --optimize-rerank
if errorlevel 1 goto :error

echo Monthly update completed successfully.
exit /b 0

:error
echo Monthly update failed.
exit /b 1
