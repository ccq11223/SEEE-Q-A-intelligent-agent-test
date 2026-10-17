@echo off
rem Run pipeline from any directory without cd first.
rem
rem Usage:
rem   & "D:\path\to\agent-code\run-pipeline.cmd" golden --worksheet
rem   & "D:\path\to\agent-code\run-pipeline.cmd" golden --check
rem
rem %~dp0 is the folder containing this script. We change into it first,
rem so "python -m pipeline" can always locate the package.
rem
rem NOTE: keep this file ASCII-only. cmd.exe reads .cmd files in the OEM
rem code page, so UTF-8 Chinese comments corrupt the following lines.

setlocal
cd /d "%~dp0"
python -m pipeline %*
exit /b %ERRORLEVEL%
