import torch
import torch.nn as nn
import timm

class SpatialEncoder(nn.Module):
    """
    Spatial feature encoder using pretrained backbones (e.g., ConvNeXt, EfficientNet).
    Extracts high-level semantic and texture features from the RGB image.
    """
    def __init__(
        self, 
        backbone_name: str = "convnext_tiny", 
        pretrained: bool = True,
        freeze_epochs: int = 0
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.freeze_epochs = freeze_epochs
        
        # Load backbone without classification head
        self.model = timm.create_model(
            backbone_name, 
            pretrained=pretrained, 
            num_classes=0,  # Remove classifier
            global_pool=''  # We want the raw feature map for spatial attention/Grad-CAM
        )
        
        # Get output feature channels
        self.num_features = self.model.num_features
        
        # Adaptive pooling to ensure consistent output size
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor, return_spatial: bool = False) -> torch.Tensor:
        """
        Args:
            x: Input image tensor (B, C, H, W)
            return_spatial: If true, returns the 2D feature map before pooling (for Grad-CAM)
        """
        features = self.model(x)
        
        if return_spatial:
            return features
            
        # (B, C, H, W) -> (B, C, 1, 1) -> (B, C)
        pooled = self.pool(features).flatten(1)
        return pooled
        
    def freeze(self):
        """Freezes the backbone parameters."""
        for param in self.model.parameters():
            param.requires_grad = False
            
    def unfreeze(self):
        """Unfreezes the backbone parameters."""
        for param in self.model.parameters():
            param.requires_grad = True
