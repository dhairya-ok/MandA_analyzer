param(
	[string]$HostIp = "0.0.0.0",
	[int]$Port = 8000
)

$ErrorActionPreference = "Stop"
Push-Location $PSScriptRoot\..

if (-Not (Test-Path .\.venv\Scripts\Activate.ps1)) {
	python -m venv .venv
}

. .\.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt

$env:PYTHONUNBUFFERED = "1"
uvicorn backend.api.main:app --host $HostIp --port $Port

Pop-Location
