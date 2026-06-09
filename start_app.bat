@echo off
setlocal

set "APP_DIR=%~dp0"
set "VENV_ACTIVATE=C:\venvs\subocr38\Scripts\activate.bat"

if not exist "%APP_DIR%app.py" (
    echo app.py not found: "%APP_DIR%app.py"
    pause
    exit /b 1
)

if not exist "%VENV_ACTIVATE%" (
    echo Virtual environment activate script not found: "%VENV_ACTIVATE%"
    pause
    exit /b 1
)

cd /d "%APP_DIR%"
call "%VENV_ACTIVATE%"
python -m streamlit run app.py
