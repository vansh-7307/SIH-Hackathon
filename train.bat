@echo off
title SignalScope Model Training
echo ========================================================
echo          SIGNALSCOPE - MODEL TRAINING
echo ========================================================
echo.
echo This will train the SignalScope model using your dataset in D:\vivek.
echo Since you are training on a CPU, this will take some time.
echo The model will automatically save the best weights to model/weights/best_model.pt
echo.
echo Press any key to begin training...
pause >nul

python scripts/train.py --train_dir "D:\vivek\train" --test_dir "D:\vivek\test" --config configs/train.yaml

echo.
echo ========================================================
echo TRAINING COMPLETE!
echo You can now run start_api.bat to use your highly accurate model!
echo ========================================================
pause
