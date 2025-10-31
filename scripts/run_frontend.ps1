param(
	[string]$ApiBase = "http://localhost:8000",
	[int]$Port = 5173
)

$ErrorActionPreference = "Stop"
Push-Location $PSScriptRoot\..\frontend

if (-Not (Test-Path .\node_modules)) {
	npm install --no-audit --no-fund
}

$env:VITE_API_BASE_URL = $ApiBase
npm run dev -- --host 0.0.0.0 --port $Port

Pop-Location
