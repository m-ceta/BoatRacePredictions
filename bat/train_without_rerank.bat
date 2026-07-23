@echo off
setlocal EnableDelayedExpansion

call "%~dp0_common.bat"
call "%~dp0_common.bat" :activate_conda
if errorlevel 1 exit /b %errorlevel%

if not defined PIPELINE_STATE_DIR set "PIPELINE_STATE_DIR=%CD%\.gcloud_pipeline_state_without_rerank"
if not defined MAX_RACES set "MAX_RACES=0"
if not defined EVAL_MAX_RACES set "EVAL_MAX_RACES=10000"

if exist "%PIPELINE_STATE_DIR%" rmdir /s /q "%PIPELINE_STATE_DIR%"
mkdir "%PIPELINE_STATE_DIR%"

call "%~dp0_common.bat" :run_step "[1/6] drive_mount" call bat\drive_mount.bat
if errorlevel 1 exit /b %errorlevel%
type nul > "%PIPELINE_STATE_DIR%\01_drive_mount.done"

call "%~dp0_common.bat" :run_step "[2/6] zip_update_local" call bat\zip_update_local.bat
if errorlevel 1 exit /b %errorlevel%
type nul > "%PIPELINE_STATE_DIR%\02_zip_update_local.done"

call "%~dp0_common.bat" :run_step "[3/6] build" boatrace-build --rowdata rowdata --output data/processed
if errorlevel 1 exit /b %errorlevel%
type nul > "%PIPELINE_STATE_DIR%\03_build.done"

call "%~dp0_common.bat" :run_step "[4/6] train without rerank optimization" boatrace-train --config configs/train.yaml
if errorlevel 1 exit /b %errorlevel%
call "%~dp0_common.bat" :run_step "[4/6] train trifecta v2/v3" boatrace-train-trifecta-v2 --config configs/train.yaml --max-races %MAX_RACES% --eval-max-races %EVAL_MAX_RACES%
if errorlevel 1 exit /b %errorlevel%
type nul > "%PIPELINE_STATE_DIR%\04_train.done"

call "%~dp0_common.bat" :run_step "[5/6] full trifecta evaluation" boatrace-eval-trifecta-full --config configs/train.yaml
if errorlevel 1 exit /b %errorlevel%
type nul > "%PIPELINE_STATE_DIR%\05_full_trifecta_evaluation.done"

call "%~dp0_common.bat" :run_step "[6/6] zip_upload" call bat\zip_upload.bat
if errorlevel 1 exit /b %errorlevel%
type nul > "%PIPELINE_STATE_DIR%\06_zip_upload.done"

echo train_without_rerank.bat completed successfully.
exit /b 0
