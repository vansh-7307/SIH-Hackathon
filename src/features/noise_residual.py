import torch
import torch.nn as nn
import torch.nn.functional as F

class HighPassFilter(nn.Module):
    """
    Applies fixed high-pass filters (SRM-like) to extract noise residuals.
    This helps in detecting local synthesis artifacts and texture inconsistencies.
    """
    def __init__(self):
        super().__init__()
        # SRM basic filter for detecting local dependencies
        filter_kernel = torch.tensor([
            [-1,  2, -1],
            [ 2, -4,  2],
            [-1,  2, -1]
        ], dtype=torch.float32) / 4.0
        
        # Reshape for conv2d
        self.kernel = filter_kernel.view(1, 1, 3, 3)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convert to grayscale
        if x.shape[1] == 3:
            weights = torch.tensor([0.299, 0.587, 0.114], device=x.device).view(1, 3, 1, 1)
            x = (x * weights).sum(dim=1, keepdim=True)
            
        # Apply filter
        weight = self.kernel.to(x.device)
        residual = F.conv2d(x, weight, padding=1)
        return residual
