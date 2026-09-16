@echo off
REM  Stop whatever is listening on the web app's port.
REM
REM  `npm run dev` spawns a child node process, so killing the npm window alone
REM  can leave the port held. This finds the owner of the listening socket and
REM  kills that process tree instead.

setlocal enabledelayedexpansion
set "PORT=3000"

echo.
echo   Stopping the workbench web app on port %PORT% ...

set "FOUND="
for /f "tokens=5" %%P in ('netstat -ano ^| findstr /r /c:"LISTENING" ^| findstr /c:":%PORT% "') do (
  if not "%%P"=="0" (
    set "FOUND=1"
    echo   killing pid %%P
    taskkill /pid %%P /t /f >nul 2>&1
  )
)

if not defined FOUND (
  echo   Nothing was listening on port %PORT%.
) else (
  echo   Stopped.
)

echo.
ping -n 3 127.0.0.1 >nul 2>&1
exit /b 0
