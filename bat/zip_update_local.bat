@echo off
setlocal

call "%~dp0_common.bat"
call "%~dp0_common.bat" :activate_conda
if errorlevel 1 exit /b %errorlevel%

if not defined DRIVE_PACKAGE_DIR set "DRIVE_PACKAGE_DIR=%USERPROFILE%\gdrive\gcolab_workdir\btp"
if not defined RESTORE_ROWDATA set "RESTORE_ROWDATA=1"
if not defined RESTORE_DATA set "RESTORE_DATA=1"
if not defined RESTORE_ARTIFACTS set "RESTORE_ARTIFACTS=1"
if not defined UPDATE_ROWDATA set "UPDATE_ROWDATA=1"

if not exist "%DRIVE_PACKAGE_DIR%" (
    echo Drive package directory was not found: %DRIVE_PACKAGE_DIR%
    exit /b 1
)

set RESTORE_ARGS=--project-root . --source-dir "%DRIVE_PACKAGE_DIR%"
if "%RESTORE_ROWDATA%"=="0" set "RESTORE_ARGS=%RESTORE_ARGS% --skip-rowdata"
if "%RESTORE_DATA%"=="0" set "RESTORE_ARGS=%RESTORE_ARGS% --skip-data"
if "%RESTORE_ARTIFACTS%"=="0" set "RESTORE_ARGS=%RESTORE_ARGS% --skip-artifacts"

call "%~dp0_common.bat" :run_step "[1/2]" boatrace-package-restore-local %RESTORE_ARGS%
if errorlevel 1 exit /b %errorlevel%

if "%UPDATE_ROWDATA%"=="1" (
    call "%~dp0_common.bat" :run_step "[2/2]" boatrace-backfill-rowdata --rowdata rowdata
    if errorlevel 1 exit /b %errorlevel%
) else (
    echo [2/2] skipped rowdata update
)

echo zip_update_local.bat completed successfully.
exit /b 0
