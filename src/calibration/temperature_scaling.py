import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

class TemperatureScaler(nn.Module):
    """
    Applies temperature scaling to calibrate model probabilities.
    Learns a single parameter (temperature) on a validation set.
    """
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

    def fit(self, logits: torch.Tensor, labels: torch.Tensor, lr: float = 0.01, max_iter: int = 500):
        """
        Fits the temperature parameter using NLL loss on a validation set.
        """
        nll_criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        
        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(self(logits).squeeze(), labels.float())
            loss.backward()
            return loss
            
        optimizer.step(eval)
        return self.temperature.item()

def calibrate_probabilities(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Applies a known temperature to logits and returns probabilities."""
    scaled_logits = logits / temperature
    return torch.sigmoid(scaled_logits)
