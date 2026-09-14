import torch
import torch.nn as nn
from src.features.frequency import extract_frequency_spectrum

class FrequencyEncoder(nn.Module):
    """
    CNN-based encoder that processes the 2D frequency spectrum.
    Captures generator-specific spectral artifacts (e.g., checkerboard patterns).
    """
    def __init__(self, out_features: int = 256):
        super().__init__()
        self.out_features = out_features
        
        self.encoder = nn.Sequential(
            # Input: 1 channel (grayscale log magnitude spectrum)
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, out_features, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_features),
            nn.ReLU(inplace=True),
            
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
    def forward(self, x: torch.Tensor, return_spatial: bool = False) -> torch.Tensor:
        """
        x: Raw image tensor (B, C, H, W)
        """
        # Extract spectrum
        spectrum = extract_frequency_spectrum(x)
        
        # Forward pass
        features = self.encoder(spectrum)
        
        if return_spatial:
            return spectrum # For explainability heatmap
            
        return features.flatten(1)
