@echo off
echo ===================================================
echo       NeoWatch-OS Dashboard Baslatiliyor...
echo ===================================================

REM Sanal ortam kontrolu
if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
) else if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
)

REM Gerekli paketleri kontrol et ve Streamlit'i baslat
echo Streamlit uygulamasi baslatiliyor...
streamlit run app.py

pause
