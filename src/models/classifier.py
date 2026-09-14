import torch
import torch.nn as nn
from src.models.spatial_encoder import SpatialEncoder
from src.models.frequency_encoder import FrequencyEncoder
from src.models.artifact_encoder import ArtifactEncoder
from src.models.fusion import AttentionFusion

class SignalScopeModel(nn.Module):
    """
    Full SignalScope architecture integrating Spatial, Frequency, and Artifact branches.
    """
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        
        # 1. Spatial Branch
        self.spatial = SpatialEncoder(
            backbone_name=config.get('spatial_backbone', 'convnext_tiny'),
            pretrained=config.get('pretrained', True)
        )
        
        feature_dims = [self.spatial.num_features]
        self.use_freq = config.get('use_frequency', True)
        self.use_art = config.get('use_artifact', True)
        
        # 2. Frequency Branch
        if self.use_freq:
            self.freq = FrequencyEncoder(out_features=256)
            feature_dims.append(256)
            
        # 3. Artifact Branch
        if self.use_art:
            self.artifact = ArtifactEncoder(out_features=256)
            feature_dims.append(256)
            
        # 4. Fusion Layer
        self.fusion = AttentionFusion(feature_dims=feature_dims, hidden_dim=512)
        
        # 5. Classification Head
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)  # Binary classification (Real vs AI)
        )

    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        features = []
        
        # Spatial features
        sp_feat = self.spatial(x)
        features.append(sp_feat)
        
        # Frequency features
        if self.use_freq:
            fr_feat = self.freq(x)
            features.append(fr_feat)
            
        # Artifact features
        if self.use_art:
            ar_feat = self.artifact(x)
            features.append(ar_feat)
            
        # Fuse
        fused = self.fusion(features)
        
        # Classify
        logits = self.classifier(fused)
        
        if return_features:
            return logits, {
                'spatial': sp_feat,
                'fused': fused
            }
            
        return logits
