@echo off
REM ===========================================================================
REM  AI Data Science Workbench - start the web app
REM
REM  Double-click this file. It installs dependencies on first run, starts the
REM  Next.js server, waits for the port to answer, and opens your browser.
REM
REM  Close this window (or press Ctrl+C) to stop the server.
REM ===========================================================================

setlocal enabledelayedexpansion
title AI Data Science Workbench

set "ROOT=%~dp0"
set "APP=%ROOT%webapp"
set "PORT=3000"
set "URL=http://localhost:%PORT%"

echo.
echo   AI Data Science Workbench
echo   -------------------------
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

REM --- already running? ------------------------------------------------------
REM  Reuse an existing server rather than failing on a busy port.
netstat -ano | findstr /r /c:"LISTENING" | findstr /c:":%PORT% " >nul 2>&1
if not errorlevel 1 (
  echo   A server is already listening on port %PORT%.
  echo   Opening %URL%
  start "" "%URL%"
  echo.
  echo   Nothing else to do. This window can be closed.
  call :sleep 5
  exit /b 0
)

REM --- start -----------------------------------------------------------------
echo   Starting the server on port %PORT% ...
pushd "%APP%"
start "workbench-webapp" /min cmd /c "npm run dev"
popd

REM  Poll the port instead of sleeping a fixed guess: a cold start can take
REM  15 seconds, a warm one under 3.
set /a TRIES=0
:waitloop
set /a TRIES+=1
call :sleep 1
netstat -ano | findstr /r /c:"LISTENING" | findstr /c:":%PORT% " >nul 2>&1
if not errorlevel 1 goto ready
if !TRIES! geq 60 (
  echo.
  echo   [X] The server did not come up within 60 seconds.
  echo       Check the minimised "workbench-webapp" window for the error.
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
echo    The server runs in a separate minimised window.
echo    Close that window to stop it, or run stop-webapp.bat
echo   ---------------------------------------------------------------
echo.
call :sleep 8
exit /b 0

REM  `timeout` refuses to run when stdin is redirected, which is exactly what
REM  happens when this file is launched by a script rather than double-clicked.
REM  `ping` to the loopback is the portable one-second-per-hop equivalent.
:sleep
ping -n %~1 127.0.0.1 >nul 2>&1
exit /b 0
