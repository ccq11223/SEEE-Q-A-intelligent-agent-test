@echo off
setlocal
cd /d "%~dp0"
where pwsh >nul 2>nul
if not errorlevel 1 goto usepwsh
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0stop-public.ps1"
goto done
:usepwsh
pwsh -NoProfile -ExecutionPolicy Bypass -File "%~dp0stop-public.ps1"
:done
pause
