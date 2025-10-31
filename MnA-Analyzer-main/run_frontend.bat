@echo off
setlocal
cd /d %~dp0\frontend
if not exist node_modules (npm install --no-audit --no-fund)
set VITE_API_BASE_URL=http://localhost:8000
npm run dev -- --host 0.0.0.0 --port 5173
endlocal
