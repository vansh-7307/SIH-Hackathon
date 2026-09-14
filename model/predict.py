import argparse
import yaml
import json
from src.inference.predictor import SignalScopePredictor

def main():
    parser = argparse.ArgumentParser(description="SignalScope Inference")
    parser.add_argument("--image", required=True, help="Path to image file")
    parser.add_argument("--caption", default=None, help="Optional text caption")
    parser.add_argument("--config", default="configs/inference.yaml", help="Path to config")
    parser.add_argument("--checkpoint", default="model/weights/best_model.pt", help="Path to weights")
    parser.add_argument("--demo", action="store_true", help="Run in interactive demo mode")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    predictor = SignalScopePredictor(config, args.checkpoint)
    
    print(f"Analyzing: {args.image}...")
    result = predictor.predict(args.image, args.caption)
    
    if args.demo:
        print("\n" + "="*50)
        print("SIGNALSCOPE - HACKATHON DEMO".center(50))
        print("="*50)
        print(f"Verdict: {result['verdict']}")
        print(f"Confidence: {result['confidence'] * 100:.1f}%")
        print("\nEXPLANATION:")
        print(result['explanation'])
        print("\nROBUSTNESS:")
        for k, v in result['robustness'].items():
            print(f"- {k}: {v}")
        print("="*50 + "\n")
    else:
        print(json.dumps(result, indent=2))
        
if __name__ == "__main__":
    main()
