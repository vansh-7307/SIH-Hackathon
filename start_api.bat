@echo off
title SignalScope Setup and Server
echo ========================================================
echo          SIGNALSCOPE - FIRST TIME SETUP
echo ========================================================
echo.
echo Installing required Machine Learning libraries...
echo (This may take a few minutes as PyTorch is massive)
echo PLEASE DO NOT CLOSE THIS WINDOW
echo.
pip install -r requirements.txt
echo.
echo ========================================================
echo          STARTING SIGNALSCOPE API
echo ========================================================
echo.
echo The API will be available at:
echo 1. From this laptop:    http://localhost:8000/docs
echo 2. From your phone/LAN: http://10.108.23.115:8000/docs
echo.
echo Waiting for server to start...
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
pause
