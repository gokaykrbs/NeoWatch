@echo off
cd /d "%~dp0"
echo ===================================================
echo       NeoWatch-OS Dashboard Baslatiliyor...
echo ===================================================

if exist ".venv\Scripts\python.exe" (
    echo Python sanal ortami (.venv) kullaniliyor...
    ".venv\Scripts\python.exe" -m streamlit run app.py
) else if exist "venv\Scripts\python.exe" (
    echo Python sanal ortami (venv) kullaniliyor...
    "venv\Scripts\python.exe" -m streamlit run app.py
) else (
    echo Global Python ile baslatiliyor...
    python -m streamlit run app.py
)

pause
