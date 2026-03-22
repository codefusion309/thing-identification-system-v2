@echo off
echo ============================================
echo  Thing Identification System - Backend
echo  Python 3.11 + CPU only
echo ============================================
echo.

echo [INFO] Using Python:
python --version

python -c "import fastapi, torch, torchvision" >nul 2>&1
if errorlevel 1 (
    echo [INFO] Installing dependencies from offline_packages...
    pip install --no-index --find-links=./offline_packages -r requirements.txt
    pip install --no-index --find-links=./offline_packages -r requirements_torch.txt
    if errorlevel 1 (
        echo [ERROR] Failed to install dependencies.
        pause
        exit /b 1
    )
)

if not exist "data" mkdir data
if not exist "saved_model" mkdir saved_model

echo.
echo [INFO] Starting server at http://127.0.0.1:5000
echo [INFO] Press Ctrl+C to stop.
echo.
python main.py
pause