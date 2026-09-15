import argparse
import yaml
import json
import time
import torch
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.inference.predictor import SignalScopePredictor

def main():
    parser = argparse.ArgumentParser(description="SignalScope Inference")
    parser.add_argument("--image", required=True, help="Path to image file")
    parser.add_argument("--caption", default=None, help="Optional text caption")
    parser.add_argument("--config", default="configs/inference.yaml", help="Path to config")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    model_path = config.get('model_path', 'artifacts/models/signalscope_best.pt')
    if not os.path.exists(model_path):
        model_path = 'model/weights/best_model.pt' # fallback
        
    predictor = SignalScopePredictor(config, model_path)
    
    start_time = time.time()
    result = predictor.predict(args.image, args.caption)
    latency_ms = (time.time() - start_time) * 1000
    
    # We can inject temperature scaling if the predictor didn't apply it natively
    temp = config.get('calibration_temp', 1.0)
    
    # Extract raw probability
    raw_prob = result.get('raw_prob', result.get('confidence', 0.5))
    if result.get('label') == 0:
        raw_prob = 1.0 - raw_prob
        
    import math
    # Reverse sigmoid to get logit
    try:
        logit = math.log(raw_prob / (1 - raw_prob + 1e-9))
    except:
        logit = 0
        
    calibrated_prob = 1 / (1 + math.exp(-logit / temp))
    
    is_fake = calibrated_prob > 0.5
    verdict = "AI-GENERATED" if is_fake else "REAL"
    final_conf = calibrated_prob if is_fake else (1 - calibrated_prob)
    
    output = {
        "verdict": verdict,
        "raw_probability": raw_prob,
        "calibrated_probability": calibrated_prob,
        "confidence": final_conf,
        "evidence": result.get('explanation', "Frequency and spatial artifacts analyzed"),
        "model_version": model_path,
        "inference_time_ms": latency_ms
    }
    
    print(json.dumps(output, indent=2))
        
if __name__ == "__main__":
    main()
