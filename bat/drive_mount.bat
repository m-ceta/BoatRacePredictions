@echo off
setlocal

call "%~dp0_common.bat"

if not defined RCLONE_REMOTE_PATH set "RCLONE_REMOTE_PATH=gdrive:"
if not defined GDRIVE_MOUNT_DIR set "GDRIVE_MOUNT_DIR=%USERPROFILE%\gdrive"

where rclone >nul 2>nul
if errorlevel 1 (
    echo rclone was not found. Install and configure rclone first.
    echo On Windows, rclone mount also requires WinFsp.
    echo Then run: rclone config
    exit /b 1
)

if not exist "%GDRIVE_MOUNT_DIR%" mkdir "%GDRIVE_MOUNT_DIR%"

mountvol "%GDRIVE_MOUNT_DIR%" /L >nul 2>nul
if not errorlevel 1 (
    echo Google Drive is already mounted: %GDRIVE_MOUNT_DIR%
    exit /b 0
)

echo Mounting %RCLONE_REMOTE_PATH% -^> %GDRIVE_MOUNT_DIR%
start "rclone Google Drive mount" /min rclone mount "%RCLONE_REMOTE_PATH%" "%GDRIVE_MOUNT_DIR%" --vfs-cache-mode writes --dir-cache-time 1h --poll-interval 1m

echo Google Drive mount started: %GDRIVE_MOUNT_DIR%
echo Package directory default: %GDRIVE_MOUNT_DIR%\gcolab_workdir\btp
exit /b 0
