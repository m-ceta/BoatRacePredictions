@echo off
setlocal

call "%~dp0_common.bat"

if not defined RCLONE_REMOTE_PATH set "RCLONE_REMOTE_PATH=gdrive:"
if not defined GDRIVE_MOUNT_DIR set "GDRIVE_MOUNT_DIR=%USERPROFILE%\gdrive"
if not defined DRIVE_PACKAGE_DIR set "DRIVE_PACKAGE_DIR=%GDRIVE_MOUNT_DIR%\gcolab_workdir\btp"
if not defined GDRIVE_MOUNT_WAIT_SECONDS set "GDRIVE_MOUNT_WAIT_SECONDS=90"
if not defined WAIT_FOR_DRIVE_PACKAGE set "WAIT_FOR_DRIVE_PACKAGE=0"
if not defined RCLONE_MOUNT_LOG set "RCLONE_MOUNT_LOG=%TEMP%\boatrace_rclone_mount.log"

where rclone >nul 2>nul
if errorlevel 1 (
    echo rclone was not found. Install and configure rclone first.
    echo On Windows, rclone mount also requires WinFsp.
    echo Then run: rclone config
    exit /b 1
)

rclone about "%RCLONE_REMOTE_PATH%" >nul 2>nul
if errorlevel 1 (
    echo rclone remote is not accessible: %RCLONE_REMOTE_PATH%
    echo Configure it with: rclone config
    exit /b 1
)

mountvol "%GDRIVE_MOUNT_DIR%" /L >nul 2>nul
if not errorlevel 1 (
    echo Google Drive is already mounted: %GDRIVE_MOUNT_DIR%
    exit /b 0
)

if /I "%WAIT_FOR_DRIVE_PACKAGE%"=="1" (
    call :check_package_ready
    if not errorlevel 1 (
        echo Google Drive package files are available: %DRIVE_PACKAGE_DIR%
        exit /b 0
    )
)

call :prepare_mount_point
if errorlevel 1 exit /b %errorlevel%

echo Mounting %RCLONE_REMOTE_PATH% -^> %GDRIVE_MOUNT_DIR%
start "rclone Google Drive mount" /min rclone mount "%RCLONE_REMOTE_PATH%" "%GDRIVE_MOUNT_DIR%" --vfs-cache-mode writes --dir-cache-time 1h --poll-interval 1m --log-file "%RCLONE_MOUNT_LOG%" --log-level INFO

echo Google Drive mount started: %GDRIVE_MOUNT_DIR%
echo rclone mount log: %RCLONE_MOUNT_LOG%

if /I not "%WAIT_FOR_DRIVE_PACKAGE%"=="1" (
    echo Package directory default: %DRIVE_PACKAGE_DIR%
    exit /b 0
)

echo Waiting for package zip files: %DRIVE_PACKAGE_DIR%
set /a WAITED_SECONDS=0

:wait_for_package
call :check_package_ready
if not errorlevel 1 (
    echo Google Drive mounted: %GDRIVE_MOUNT_DIR%
    echo Package directory default: %DRIVE_PACKAGE_DIR%
    exit /b 0
)
if %WAITED_SECONDS% GEQ %GDRIVE_MOUNT_WAIT_SECONDS% (
    echo Google Drive package files did not become ready within %GDRIVE_MOUNT_WAIT_SECONDS% seconds.
    echo Expected one of:
    echo   %DRIVE_PACKAGE_DIR%\rowdata.zip
    echo   %DRIVE_PACKAGE_DIR%\data.zip
    echo   %DRIVE_PACKAGE_DIR%\artifacts.zip
    echo Check that rclone remote "%RCLONE_REMOTE_PATH%" is configured, WinFsp is installed, and the package zips exist.
    exit /b 1
)
set /a WAITED_SECONDS+=1
timeout /t 1 /nobreak >nul
goto :wait_for_package

:check_package_ready
if exist "%DRIVE_PACKAGE_DIR%\rowdata.zip" exit /b 0
if exist "%DRIVE_PACKAGE_DIR%\data.zip" exit /b 0
if exist "%DRIVE_PACKAGE_DIR%\artifacts.zip" exit /b 0
exit /b 1

:prepare_mount_point
for %%P in ("%GDRIVE_MOUNT_DIR%") do set "GDRIVE_MOUNT_PARENT=%%~dpP"
if not exist "%GDRIVE_MOUNT_PARENT%" mkdir "%GDRIVE_MOUNT_PARENT%"
if not exist "%GDRIVE_MOUNT_DIR%" exit /b 0
dir /a /b "%GDRIVE_MOUNT_DIR%" 2>nul | findstr . >nul
if errorlevel 1 (
    rmdir "%GDRIVE_MOUNT_DIR%"
    if exist "%GDRIVE_MOUNT_DIR%" (
        echo Failed to remove empty mount point directory: %GDRIVE_MOUNT_DIR%
        exit /b 1
    )
    exit /b 0
)
echo Mount point path already exists and is not empty: %GDRIVE_MOUNT_DIR%
echo Choose a different GDRIVE_MOUNT_DIR or move/delete the existing directory before mounting.
exit /b 1
