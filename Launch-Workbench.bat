@echo off
REM ===========================================================================
REM  AI Data Science Workbench - launch the site (production)
REM
REM  Double-click this file. It installs dependencies on first run, builds the
REM  site if anything changed since the last build, starts the production
REM  server, and opens your browser.
REM
REM  Production mode, not dev: pages are pre-compiled, so they open instantly
REM  instead of building on first visit. Use start-webapp.bat if you want the
REM  dev server with hot reload.
REM
REM  Usage:
REM    Launch-Workbench.bat            start the site and open it
REM    Launch-Workbench.bat /rebuild   force a fresh build first
REM    Launch-Workbench.bat /stop      stop the running site
REM ===========================================================================

setlocal enabledelayedexpansion
title AI Data Science Workbench

set "ROOT=%~dp0"
set "APP=%ROOT%webapp"
set "PORT=3000"
set "URL=http://localhost:%PORT%"

if /i "%~1"=="/stop" goto :stop

echo.
echo   AI Data Science Workbench
echo   =========================
echo.

REM --- prerequisites ---------------------------------------------------------
where node >nul 2>&1
if errorlevel 1 (
  echo   [X] Node.js was not found on PATH.
  echo       Install it from https://nodejs.org and run this file again.
  echo.
  pause
  exit /b 1
)

where python >nul 2>&1
if errorlevel 1 (
  echo   [!] Python was not found on PATH.
  echo       The site will start, but uploading a dataset will fail - the
  echo       analysis runs through Python. Install Python 3.12+ to fix it.
  echo.
)

if not exist "%APP%\package.json" (
  echo   [X] Could not find "%APP%\package.json".
  echo       Keep this file next to the webapp folder.
  echo.
  pause
  exit /b 1
)

REM --- already running? ------------------------------------------------------
REM  Reuse an existing server rather than failing on a busy port.
netstat -ano | findstr /r /c:"LISTENING" | findstr /c:":%PORT% " >nul 2>&1
if not errorlevel 1 (
  echo   A server is already listening on port %PORT%.
  echo   Opening %URL%
  start "" "%URL%"
  echo.
  call :sleep 4
  exit /b 0
)

REM --- dependencies ----------------------------------------------------------
if not exist "%APP%\node_modules" (
  echo   First run - installing dependencies. This takes a few minutes.
  echo.
  pushd "%APP%"
  call npm install
  set "INSTALL_FAILED=!errorlevel!"
  popd
  if not "!INSTALL_FAILED!"=="0" (
    echo.
    echo   [X] npm install failed. Scroll up for the reason.
    pause
    exit /b 1
  )
  echo.
)

REM --- build, but only when it would change anything --------------------------
set "MUST_BUILD=0"
if /i "%~1"=="/rebuild" (
  echo   Forcing a fresh build.
  set "MUST_BUILD=1"
) else (
  pushd "%APP%"
  node scripts\needs-build.mjs
  if errorlevel 1 set "MUST_BUILD=1"
  popd
)

if "!MUST_BUILD!"=="1" (
  echo   Building the site. First build takes about a minute.
  echo.
  pushd "%APP%"
  call npm run build
  set "BUILD_FAILED=!errorlevel!"
  popd
  if not "!BUILD_FAILED!"=="0" (
    echo.
    echo   [X] The build failed. Scroll up for the error.
    pause
    exit /b 1
  )
  echo.
)

REM --- start -----------------------------------------------------------------
echo   Starting the site on port %PORT% ...
pushd "%APP%"
start "workbench-site" /min cmd /c "npm run start"
popd

REM  Poll the port rather than guessing a fixed wait.
set /a TRIES=0
:waitloop
set /a TRIES+=1
call :sleep 1
netstat -ano | findstr /r /c:"LISTENING" | findstr /c:":%PORT% " >nul 2>&1
if not errorlevel 1 goto ready
if !TRIES! geq 60 (
  echo.
  echo   [X] The site did not come up within 60 seconds.
  echo       Check the minimised "workbench-site" window for the error.
  echo.
  pause
  exit /b 1
)
goto waitloop

:ready
echo   Ready. Opening %URL%
echo.
start "" "%URL%"

echo   ---------------------------------------------------------------
echo    The site runs in a separate minimised window.
echo    Stop it with:  Launch-Workbench.bat /stop
echo   ---------------------------------------------------------------
echo.
call :sleep 8
exit /b 0

REM ---------------------------------------------------------------------------
:stop
echo.
echo   Stopping the site on port %PORT% ...
set "FOUND="
for /f "tokens=5" %%P in ('netstat -ano ^| findstr /r /c:"LISTENING" ^| findstr /c:":%PORT% "') do (
  if not "%%P"=="0" (
    set "FOUND=1"
    echo   killing pid %%P
    taskkill /pid %%P /t /f >nul 2>&1
  )
)
if not defined FOUND (echo   Nothing was listening on port %PORT%.) else (echo   Stopped.)
echo.
call :sleep 3
exit /b 0

REM  `timeout` refuses to run when stdin is redirected, which is what happens
REM  when this file is launched by a script rather than double-clicked.
REM  `ping` to the loopback is the portable one-second-per-hop equivalent.
:sleep
ping -n %~1 127.0.0.1 >nul 2>&1
exit /b 0
