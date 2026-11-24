@echo off
setlocal enabledelayedexpansion
rem Python prints UTF-8 whatever the console's code page (a Chinese Windows console is GBK)
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

set "APP_ROOT=%~dp0"
if "%APP_ROOT:~-1%"=="\" set "APP_ROOT=%APP_ROOT:~0,-1%"
set "UPDATE_DIR=%APP_ROOT%\update"
set "APPLIED_DIR=%UPDATE_DIR%\applied"
set "BACKUP_DIR=%APP_ROOT%\backups"
set "VENV_PY=%APP_ROOT%\.venv\Scripts\python.exe"
if not exist "%VENV_PY%" (
  echo ShotSpace is not installed yet: run setup.bat first.
  exit /b 1
)

set "CMD=%~1"
if "%CMD%"=="" set "CMD=start"
if /I "%CMD%"=="start" goto :start
if /I "%CMD%"=="stop" goto :stop
if /I "%CMD%"=="status" goto :status
if /I "%CMD%"=="update" goto :update
if /I "%CMD%"=="rollback" goto :rollback
if /I "%CMD%"=="doctor" goto :doctor
if /I "%CMD%"=="passwd" goto :passwd
goto :usage

:start
set "PORT_ARG="
if not "%~2"=="" set "PORT_ARG=--port %~2"
"%VENV_PY%" -m shotspace.cli init %PORT_ARG% || exit /b 1
set "CFG_PORT=8765"
for /f "usebackq delims=" %%P in (`call "%VENV_PY%" -c "from shotspace.server_config import ConfigStore; print(ConfigStore().load().port or 8765)"`) do set "CFG_PORT=%%P"
if not "%~2"=="" set "CFG_PORT=%~2"
call :find_pid
if defined SERVER_PID (
  echo [START] ShotSpace is already running ^(PID %SERVER_PID%^) -^> http://127.0.0.1:%CFG_PORT%/
  exit /b 0
)
if exist "%APP_ROOT%\shotspace.log" if exist "%APP_ROOT%\shotspace.log.old" del /q "%APP_ROOT%\shotspace.log.old"
if exist "%APP_ROOT%\shotspace.log" ren "%APP_ROOT%\shotspace.log" shotspace.log.old
echo [START] Starting ShotSpace on port %CFG_PORT%...
rem a hidden process of its own: it keeps running when this window is closed
powershell -NoProfile -Command "Start-Process -WindowStyle Hidden -WorkingDirectory '%APP_ROOT%' -FilePath '%VENV_PY%' -ArgumentList '-m','shotspace.cli','serve','--host','0.0.0.0','--port','%CFG_PORT%' -RedirectStandardOutput '%APP_ROOT%\shotspace.log' -RedirectStandardError '%APP_ROOT%\shotspace.err.log'"
set /a TRIES=0
:wait_health
set /a TRIES+=1
"%VENV_PY%" -c "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:%CFG_PORT%/api/health', timeout=20).status == 200 else 1)" >nul 2>&1
if not errorlevel 1 goto :started
if %TRIES% geq 40 (
  echo [ERROR] ShotSpace did not answer within 2 minutes. See shotspace.log and shotspace.err.log.
  exit /b 1
)
powershell -NoProfile -Command "Start-Sleep -Seconds 2"
goto :wait_health
:started
echo [OK] ShotSpace is running -^> http://127.0.0.1:%CFG_PORT%/   (admin page: /admin)
exit /b 0

:status
call :find_pid
if defined SERVER_PID (
  echo [STATUS] running ^(PID %SERVER_PID%^)
  exit /b 0
)
echo [STATUS] not running
exit /b 1

:find_pid
set "SERVER_PID="
for /f %%Q in ('powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*shotspace.cli serve*' -and $_.ExecutablePath -like '%APP_ROOT%\*' } | Select-Object -First 1 -ExpandProperty ProcessId"') do set "SERVER_PID=%%Q"
exit /b 0

:passwd
"%VENV_PY%" -m shotspace.cli passwd
exit /b %ERRORLEVEL%

:stop
echo [STOP] Stopping ShotSpace under %APP_ROOT%...
rem the service and its GPU workers (every Python of this folder), with their child processes
powershell -NoProfile -Command ^
  "Get-CimInstance Win32_Process | Where-Object { $_.ExecutablePath -like '%APP_ROOT%\*' -and ($_.CommandLine -like '*shotspace.cli serve*' -or $_.CommandLine -like '*shotspace\solver\workers\*' -or $_.CommandLine -like '*shotspace\io\workers\*') } | ForEach-Object { taskkill /T /F /PID $_.ProcessId 2>&1 | Out-Null }"
echo [OK] Stop completed.
exit /b 0

:doctor
"%VENV_PY%" -m shotspace.cli doctor
exit /b %ERRORLEVEL%

:update
if not exist "%UPDATE_DIR%" mkdir "%UPDATE_DIR%"
if not exist "%APPLIED_DIR%" mkdir "%APPLIED_DIR%"
if not exist "%BACKUP_DIR%" mkdir "%BACKUP_DIR%"

set "PATCH_FILE="
if not "%~2"=="" (
  if exist "%UPDATE_DIR%\%~nx2" (
    set "PATCH_FILE=%UPDATE_DIR%\%~nx2"
  ) else (
    echo [REJECTED] Patch file must be placed inside "%UPDATE_DIR%\" before running update!
    exit /b 1
  )
) else (
  for /f "delims=" %%F in ('dir /b /o-d "%UPDATE_DIR%\*.tar.gz" 2^>nul') do (
    if "!PATCH_FILE!"=="" set "PATCH_FILE=%UPDATE_DIR%\%%F"
  )
)

if "%PATCH_FILE%"=="" (
  echo [ERROR] No *.tar.gz patch archive found in "%UPDATE_DIR%\".
  exit /b 1
)

echo ==^> [1/6] Verifying archive "%PATCH_FILE%"...
tar -tzf "%PATCH_FILE%" >nul || (echo [ABORT] Corrupt archive, aborting before stopping service! & exit /b 1)

for /f %%T in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "TS=%%T"
set "BKP=%BACKUP_DIR%\app_backup_%TS%.tar.gz"
echo ==^> [2/6] Backing up live code to "%BKP%"...
set "BKP_ITEMS="
for %%X in (shotspace tests install webui\dist shotspace.sh shotspace.bat setup.sh setup.bat pyproject.toml README.md config\default.json) do if exist "%APP_ROOT%\%%X" set "BKP_ITEMS=!BKP_ITEMS! %%X"
tar -czf "%BKP%" -C "%APP_ROOT%" !BKP_ITEMS!

echo ==^> [3/6] Stopping current ShotSpace instance...
call "%~f0" stop

echo ==^> [4/6] Extracting patch over "%APP_ROOT%"...
tar -xzf "%PATCH_FILE%" -C "%APP_ROOT%" || exit /b 1

echo ==^> [5/6] Running Python module import pre-check...
"%VENV_PY%" -c "import shotspace, shotspace.web.app, shotspace.core.camera_space; print('[PRECHECK OK]', shotspace.__version__)" || (
  echo [ERROR] Pre-check failed! Run "shotspace.bat rollback" to restore "%BKP%".
  exit /b 1
)

for %%A in ("%PATCH_FILE%") do set "STEM=%%~nA"
set "STEM=%STEM:.tar=%"
move /Y "%PATCH_FILE%" "%APPLIED_DIR%\%STEM%_applied_%TS%.tar.gz" >nul
echo ==^> [6/6] Restarting ShotSpace...
call "%~f0" start
exit /b 0

:rollback
set "LATEST_BKP="
for /f "delims=" %%F in ('dir /b /o-d "%BACKUP_DIR%\app_backup_*.tar.gz" 2^>nul') do (
  if "!LATEST_BKP!"=="" set "LATEST_BKP=%BACKUP_DIR%\%%F"
)
if "%LATEST_BKP%"=="" (
  echo [ERROR] No backup archive found in "%BACKUP_DIR%\".
  exit /b 1
)
tar -tzf "%LATEST_BKP%" >nul || exit /b 1
call "%~f0" stop
tar -xzf "%LATEST_BKP%" -C "%APP_ROOT%" || exit /b 1
"%VENV_PY%" -c "import shotspace, shotspace.web.app; print('[ROLLBACK PRECHECK OK]', shotspace.__version__)" || exit /b 1
call "%~f0" start
echo [OK] Rolled back to "%LATEST_BKP%".
exit /b 0

:usage
echo Usage: shotspace.bat ^<start^|stop^|status^|doctor^|passwd^|update^|rollback^>
exit /b 1
