@echo off
setlocal

call "%~dp0_common.bat"
call "%~dp0_common.bat" :activate_conda
if errorlevel 1 exit /b %errorlevel%

if not defined DRIVE_PACKAGE_DIR set "DRIVE_PACKAGE_DIR=%USERPROFILE%\gdrive\gcolab_workdir\btp"
if not defined EXPORT_ROWDATA set "EXPORT_ROWDATA=1"
if not defined EXPORT_DATA set "EXPORT_DATA=1"
if not defined EXPORT_ARTIFACTS set "EXPORT_ARTIFACTS=1"

if not exist "%DRIVE_PACKAGE_DIR%" mkdir "%DRIVE_PACKAGE_DIR%"

set EXPORT_ARGS=--project-root . --output-dir "%DRIVE_PACKAGE_DIR%"
if "%EXPORT_ROWDATA%"=="0" set "EXPORT_ARGS=%EXPORT_ARGS% --skip-rowdata"
if "%EXPORT_DATA%"=="0" set "EXPORT_ARGS=%EXPORT_ARGS% --skip-data"
if "%EXPORT_ARTIFACTS%"=="0" set "EXPORT_ARGS=%EXPORT_ARGS% --skip-artifacts"

call "%~dp0_common.bat" :run_step "[1/1]" boatrace-package-export %EXPORT_ARGS%
if errorlevel 1 exit /b %errorlevel%

echo zip_upload.bat completed successfully.
exit /b 0
