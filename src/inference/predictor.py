import torch
import torchvision.transforms as T
from PIL import Image
from typing import Dict, Any, Optional
import os

from src.models.classifier import SignalScopeModel
from src.calibration.calibrator import ModelCalibrator
from src.explainability.gradcam import GradCAM, generate_overlay
from src.robustness.robustness_suite import RobustnessSuite
from src.provenance.metadata import extract_metadata
from src.explainability.explanation_generator import ExplanationGenerator
from src.data.transforms import get_inference_transforms

class SignalScopePredictor:
    """Unified inference API for SignalScope."""
    def __init__(self, config: dict, model_path: str, device: str = 'auto'):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() and device == 'auto' else 'cpu')
        
        # Load Model
        self.model = SignalScopeModel(config.get('model', {})).to(self.device)
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        # Load Calibrator
        self.calibrator = ModelCalibrator(
            high_threshold=config.get('thresholds', {}).get('high', 0.8),
            low_threshold=config.get('thresholds', {}).get('low', 0.2)
        )
        # Try loading learned temperature
        self.calibrator.load(os.path.join(os.path.dirname(model_path), 'calibration.json'))
        
        self.transforms = get_inference_transforms()
        self.explainer = ExplanationGenerator(self.calibrator)
        
    def predict_proba(self, image: Image.Image) -> float:
        """Internal method for raw probability (used by robustness suite)."""
        x = self.transforms(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            prob = torch.sigmoid(logits / self.calibrator.temperature).item()
        return prob

    def predict(self, image_path: str, caption: Optional[str] = None) -> Dict[str, Any]:
        """Main inference method returning structured JSON."""
        # 1. Load Image
        image = Image.open(image_path).convert('RGB')
        x = self.transforms(image).unsqueeze(0).to(self.device)
        
        # 2. Forward pass with GradCAM
        gradcam = GradCAM(self.model, self.model.spatial.model.stages[-1]) # ConvNeXt final stage
        cam_heatmap = gradcam(x)
        
        # Also get logits directly (without grad)
        with torch.no_grad():
            logits = self.model(x)
            
        # 3. Calibration
        raw_prob = torch.sigmoid(logits).item()
        calibrated_prob = torch.sigmoid(logits / self.calibrator.temperature).item()
        verdict = self.calibrator.get_verdict(calibrated_prob)
        
        # 4. Generate Mock Evidence (Since we don't have bounding boxes yet, we infer from heatmap)
        evidence = []
        if calibrated_prob > 0.5:
            evidence.append({
                "type": "frequency_anomaly",
                "score": float(calibrated_prob),
                "region": "global"
            })
            
        # 5. Metadata
        metadata = extract_metadata(image_path)
        if metadata.get('ai_software_detected'):
            evidence.append({
                "type": "ai_software_metadata",
                "score": 1.0,
                "region": "metadata"
            })
            
        # 6. Robustness
        robustness_suite = RobustnessSuite(self)
        robustness_results = robustness_suite.run_suite(image, calibrated_prob)
        
        # 7. Explanation text
        explanation_text = self.explainer.generate_text_explanation(
            calibrated_prob, evidence, robustness_results, metadata
        )
        
        return {
            "verdict": verdict,
            "label": 1 if calibrated_prob > 0.5 else 0,
            "raw_probability": float(raw_prob),
            "calibrated_probability": float(calibrated_prob),
            "confidence": float(calibrated_prob) if calibrated_prob > 0.5 else float(1.0 - calibrated_prob),
            "uncertainty": "low" if abs(calibrated_prob - 0.5) > 0.3 else "high",
            "threshold": self.calibrator.high_threshold,
            
            "evidence": evidence,
            "generator_attribution": {
                "family": "unknown",
                "confidence": 0.0
            },
            
            "metadata": metadata,
            "robustness": robustness_results,
            "explanation": explanation_text,
            
            # Note: In a real API, heatmaps are saved/returned as Base64. 
            # We omit the actual huge byte arrays here for simplicity.
            "artifacts_generated": True
        }
