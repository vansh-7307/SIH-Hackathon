import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionFusion(nn.Module):
    """
    Learned attention-based fusion for multiple feature branches.
    """
    def __init__(self, feature_dims: list, hidden_dim: int = 512):
        super().__init__()
        self.num_branches = len(feature_dims)
        self.hidden_dim = hidden_dim
        
        # Project all features to same dimension
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU()
            ) for dim in feature_dims
        ])
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, features: list) -> torch.Tensor:
        """
        features: List of feature tensors from different branches
        """
        # Project all features: List of (B, hidden_dim)
        proj_feats = [proj(feat) for proj, feat in zip(self.projections, features)]
        
        # Stack: (B, num_branches, hidden_dim)
        stacked = torch.stack(proj_feats, dim=1)
        
        # Compute attention weights: (B, num_branches, 1)
        attn_logits = self.attention(stacked)
        attn_weights = F.softmax(attn_logits, dim=1)
        
        # Weighted sum: (B, hidden_dim)
        fused = torch.sum(stacked * attn_weights, dim=1)
        
        return fused
