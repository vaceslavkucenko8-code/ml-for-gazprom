@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    python -m venv .venv
    if errorlevel 1 goto error
)
.venv\Scripts\python.exe -c "import requests,pydantic,bs4,lxml,dateutil,rapidfuzz,py3langid,openpyxl,sklearn,fastapi,uvicorn,psycopg,ctranslate2,sentencepiece" >nul 2>&1
if errorlevel 1 (
    .venv\Scripts\python.exe -m pip install -r requirements.txt
    if errorlevel 1 goto error
)
.venv\Scripts\python.exe -m src.local_app
if errorlevel 1 goto error
exit /b 0
:error
echo Launch failed. Install Python 3.12 and check internet access.
pause
exit /b 1
