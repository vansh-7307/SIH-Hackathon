from typing import Dict, Any

class ExplanationGenerator:
    def __init__(self, calibrator):
        self.calibrator = calibrator

    def generate_text_explanation(
        self, 
        calibrated_prob: float, 
        evidence: list, 
        robustness: dict, 
        metadata: dict
    ) -> str:
        """Generates faithful human-readable text from evidence dictionary."""
        verdict = self.calibrator.get_verdict(calibrated_prob)
        conf_pct = int(calibrated_prob * 100)
        
        explanation = f"SignalScope assessed this image as '{verdict}' with {conf_pct}% confidence. "
        
        if verdict == "Likely AI-generated":
            reasons = []
            for ev in evidence:
                if ev['score'] > 0.6:
                    reasons.append(ev['type'].replace('_', ' '))
            
            if reasons:
                explanation += f"The system detected anomalies consistent with synthesis, specifically: {', '.join(reasons)}. "
                
        elif verdict == "Likely authentic":
            explanation += "The system found no significant evidence of AI synthesis across spatial and frequency domains. "
            
        else:
            explanation += "The conflicting or weak signals prevent a confident assessment. "
            
        if robustness.get('stable_under_jpeg', False):
            explanation += "This prediction remained stable even after common image degradations like JPEG compression. "
            
        if metadata.get('c2pa_detected', False):
            explanation += "Note: C2PA provenance credentials were detected, which may provide definitive origin information. "
            
        explanation += "Please note that this is a probabilistic assessment based on visible artifacts and should not be treated as absolute proof."
        
        return explanation
