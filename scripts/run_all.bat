@echo off
echo Running full pipeline for SignalScope...

echo ==================================================
echo 1. DATASET PREPARATION (Tiny-GenImage)
echo ==================================================
python scripts/prepare_data.py --download --limit 2000

echo ==================================================
echo 2. TRAINING BASELINE MODEL (Experiment 1)
echo ==================================================
python scripts/train.py --config configs/train.yaml --baseline

echo ==================================================
echo 3. TRAINING FULL SIGNALSCOPE MODEL (Experiment 4)
echo ==================================================
python scripts/train.py --config configs/train.yaml --experiment "Full SignalScope model"

echo ==================================================
echo 4. CLI PREDICTION TEST
echo ==================================================
:: We will test with a real and a fake image from the validation set
set test_img="data/Tiny-GenImage/validation/real/real_real_00001.jpg"
if exist %test_img% (
    python model/predict.py --image %test_img%
)

echo Pipeline complete!
