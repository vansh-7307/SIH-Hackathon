import torch
import torchvision.transforms as T
from PIL import Image, ImageFilter
import io

class RobustnessSuite:
    """
    Applies image degradations to test prediction stability.
    """
    def __init__(self, predictor):
        self.predictor = predictor
        
    def test_jpeg(self, image: Image.Image, quality: int) -> float:
        buf = io.BytesIO()
        image.save(buf, format='JPEG', quality=quality)
        buf.seek(0)
        img_jpeg = Image.open(buf).convert('RGB')
        return self.predictor.predict_proba(img_jpeg)
        
    def test_blur(self, image: Image.Image, radius: float = 1.0) -> float:
        img_blur = image.filter(ImageFilter.GaussianBlur(radius))
        return self.predictor.predict_proba(img_blur)
        
    def test_resize(self, image: Image.Image, scale: float = 0.5) -> float:
        w, h = image.size
        img_res = image.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
        # Resize back to original size for the model
        img_res = img_res.resize((w, h), Image.Resampling.LANCZOS)
        return self.predictor.predict_proba(img_res)

    def run_suite(self, image: Image.Image, original_prob: float) -> dict:
        results = {
            'original_prob': original_prob
        }
        
        # JPEG 50
        prob_jpeg_50 = self.test_jpeg(image, 50)
        results['jpeg_50_prob'] = prob_jpeg_50
        
        # Blur
        prob_blur = self.test_blur(image, 1.0)
        results['blur_prob'] = prob_blur
        
        # Verdict stability
        orig_verdict = original_prob > 0.5
        jpeg_verdict = prob_jpeg_50 > 0.5
        
        results['stable_under_jpeg'] = (orig_verdict == jpeg_verdict)
        
        return results
