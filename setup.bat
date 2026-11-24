@echo off
rem ShotSpace installer (Windows).  Double-click, or run:  setup.bat
rem Everything goes inside this folder (.tools, third_party): nothing is installed system-wide, PATH and the
rem registry are not changed.  Re-running continues where the last run stopped.
setlocal
rem Python prints UTF-8 whatever the console's code page (a Chinese Windows console is GBK)
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%ROOT%\install\bootstrap.ps1" %*
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
  echo.
  echo Setup stopped with code %RC%.
)
if "%SHOTSPACE_NO_PAUSE%"=="" pause
exit /b %RC%
