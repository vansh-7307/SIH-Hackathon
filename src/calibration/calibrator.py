import json
import torch
import os
from typing import Dict, Any

class ModelCalibrator:
    def __init__(self, high_threshold: float = 0.8, low_threshold: float = 0.2):
        self.temperature = 1.0
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        
    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump({
                'temperature': self.temperature,
                'high_threshold': self.high_threshold,
                'low_threshold': self.low_threshold
            }, f)
            
    def load(self, path: str):
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
                self.temperature = data.get('temperature', 1.0)
                self.high_threshold = data.get('high_threshold', 0.8)
                self.low_threshold = data.get('low_threshold', 0.2)
                
    def get_verdict(self, calibrated_prob: float) -> str:
        if calibrated_prob >= self.high_threshold:
            return "Likely AI-generated"
        elif calibrated_prob <= self.low_threshold:
            return "Likely authentic"
        else:
            return "Uncertain"
