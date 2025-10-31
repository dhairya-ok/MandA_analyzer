@echo off
setlocal
cd /d %~dp0
if not exist .venv (python -m venv .venv)
call .venv\Scripts\activate.bat
pip install -r backend\requirements.txt
set PYTHONUNBUFFERED=1
uvicorn backend.api.main:app --host 0.0.0.0 --port 8000
endlocal
