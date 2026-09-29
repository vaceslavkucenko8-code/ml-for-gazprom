$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath '.venv/Scripts/python.exe')) {
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Install Python 3.12 and try again.' }
}
& .venv/Scripts/python.exe -c 'import requests,pydantic,bs4,lxml,dateutil,rapidfuzz,py3langid,openpyxl,sklearn,fastapi,uvicorn,psycopg' 2>$null
if ($LASTEXITCODE -ne 0) {
    & .venv/Scripts/python.exe -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed. Check internet access.' }
}
& .venv/Scripts/python.exe -m src.local_app
