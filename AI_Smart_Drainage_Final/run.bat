@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Creating Python virtual environment...
    py -3 -m venv .venv
    if errorlevel 1 (
        echo Could not create the virtual environment. Install Python 3.10 or newer.
        exit /b 1
    )
)

echo Installing or verifying requirements...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 exit /b 1

".venv\Scripts\python.exe" run.py
