@echo off
set "PROJECT_ROOT=%~dp0.."
cd /d "%PROJECT_ROOT%"

if not defined ENV_NAME set "ENV_NAME=boatrace-predictions"

if not "%~1"=="" (
    call %*
    exit /b %errorlevel%
)

goto :eof

:log_time
powershell -NoProfile -Command "Get-Date -Format 'yyyy-MM-dd HH:mm:ss JST'"
goto :eof

:run_step
set "STEP_LABEL=%~1"
set "COMMAND_LINE=%*"
set "STEP_LABEL_QUOTED="%~1""
call set "COMMAND_LINE=%%COMMAND_LINE:%STEP_LABEL_QUOTED% =%%"
set "STEP_LABEL_RAW=%~1"
call set "COMMAND_LINE=%%COMMAND_LINE:%STEP_LABEL_RAW% =%%"
for /f "delims=" %%T in ('powershell -NoProfile -Command "Get-Date -Format 'yyyy-MM-dd HH:mm:ss JST'"') do set "NOW=%%T"
echo [%NOW%] %STEP_LABEL% %COMMAND_LINE%
call %COMMAND_LINE%
set "RUN_STEP_ERRORLEVEL=%ERRORLEVEL%"
exit /b %RUN_STEP_ERRORLEVEL%

:find_conda
set "CONDA_BAT="
for /f "delims=" %%I in ('where conda.bat 2^>nul') do (
    set "CONDA_BAT=%%I"
    goto :conda_found
)
for /f "delims=" %%I in ('where conda 2^>nul') do (
    set "CONDA_BAT=%%~dpI..\condabin\conda.bat"
    goto :conda_found
)
echo Conda was not found on PATH.
exit /b 1
:conda_found
if not exist "%CONDA_BAT%" (
    echo conda.bat was not found.
    echo Expected: %CONDA_BAT%
    exit /b 1
)
exit /b 0

:activate_conda
call :find_conda
if errorlevel 1 exit /b %errorlevel%
call "%CONDA_BAT%" activate "%ENV_NAME%"
if errorlevel 1 (
    echo Failed to activate conda environment "%ENV_NAME%".
    exit /b %errorlevel%
)
exit /b 0
