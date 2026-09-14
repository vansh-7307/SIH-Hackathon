import torch
import torch.nn as nn
from src.features.noise_residual import HighPassFilter

class ArtifactEncoder(nn.Module):
    """
    CNN-based encoder processing high-frequency noise residuals.
    """
    def __init__(self, out_features: int = 256):
        super().__init__()
        self.out_features = out_features
        self.filter = HighPassFilter()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Projection layer
        self.proj = nn.Linear(128, out_features)
        
    def forward(self, x: torch.Tensor, return_spatial: bool = False) -> torch.Tensor:
        # Extract residual
        residual = self.filter(x)
        
        # Forward pass
        features = self.encoder(residual)
        
        if return_spatial:
            return residual
            
        features = features.flatten(1)
        return self.proj(features)
