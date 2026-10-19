@echo off
cd /d "%~dp0..\.."
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
if not exist data mkdir data
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 >> data\server.log 2>&1
