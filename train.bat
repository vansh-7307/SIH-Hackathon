@echo off
title SignalScope Model Training
:menu
cls
echo ========================================================
echo          SIGNALSCOPE - MODEL TRAINING MENU
echo ========================================================
echo.
echo Your dataset in D:\vivek contains over 100,000 images!
echo Since you are training on a laptop CPU (no NVIDIA GPU), 
echo training on all 100,000 images will take a very long time.
echo.
echo Please choose your training mode:
echo.
echo [1] Fast-Track Training (Takes ~15 minutes)
echo     Trains on a perfectly balanced subset of 2,500 images.
echo     Great for a quick accuracy boost before a demo.
echo.
echo [2] Full Deep Training (Leave running overnight/days)
echo     Trains on all 100,000 images for maximum possible accuracy.
echo     Only use this if you can leave your laptop on.
echo.
echo [3] Exit
echo.

set /p choice="Enter your choice (1, 2, or 3): "

if "%choice%"=="1" goto fast
if "%choice%"=="2" goto full
if "%choice%"=="3" goto eof
goto menu

:fast
cls
echo Starting Fast-Track Training...
python scripts/better_train.py
goto finish

:full
cls
echo Starting Full Deep Training (This will take a while!)...
python scripts/train.py --train_dir "D:\vivek\train" --test_dir "D:\vivek\test" --config configs/train.yaml
goto finish

:finish
echo.
echo ========================================================
echo TRAINING COMPLETE!
echo The new brain has been saved to model/weights/best_model.pt
echo You can now run start_api.bat to use your highly accurate model!
echo ========================================================
pause
