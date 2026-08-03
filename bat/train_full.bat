@echo off
setlocal EnableDelayedExpansion

call "%~dp0_common.bat"
call "%~dp0_common.bat" :activate_conda
if errorlevel 1 exit /b %errorlevel%

if not defined PIPELINE_STATE_DIR set "PIPELINE_STATE_DIR=%CD%\.gcloud_pipeline_state"

if exist "%PIPELINE_STATE_DIR%" rmdir /s /q "%PIPELINE_STATE_DIR%"
mkdir "%PIPELINE_STATE_DIR%"

set "TRAIN_ARGS=--config configs/train.yaml"
if "%SKIP_TRAIN_EVALUATION%"=="1" set "TRAIN_ARGS=%TRAIN_ARGS% --skip-evaluation"
if "%BOATRACE_TRAIN_SKIP_EVALUATION%"=="1" set "TRAIN_ARGS=%TRAIN_ARGS% --skip-evaluation"

set "WAIT_FOR_DRIVE_PACKAGE=1"
call "%~dp0_common.bat" :run_step "[1/5] drive_mount" call bat\drive_mount.bat
if errorlevel 1 exit /b %errorlevel%
type nul > "%PIPELINE_STATE_DIR%\01_drive_mount.done"

call "%~dp0_common.bat" :run_step "[2/5] zip_update_local" call bat\zip_update_local.bat
if errorlevel 1 exit /b %errorlevel%
type nul > "%PIPELINE_STATE_DIR%\02_zip_update_local.done"

call "%~dp0_common.bat" :run_step "[3/5] build" boatrace-build --rowdata rowdata --output data/processed
if errorlevel 1 exit /b %errorlevel%
type nul > "%PIPELINE_STATE_DIR%\03_build.done"

call "%~dp0_common.bat" :run_step "[4/5] train" boatrace-train %TRAIN_ARGS%
if errorlevel 1 exit /b %errorlevel%
type nul > "%PIPELINE_STATE_DIR%\04_train.done"

call "%~dp0_common.bat" :run_step "[5/5] zip_upload" call bat\zip_upload.bat
if errorlevel 1 exit /b %errorlevel%
type nul > "%PIPELINE_STATE_DIR%\05_zip_upload.done"

echo train_full.bat completed successfully.
exit /b 0
