import torch
import torch.nn as nn
import torch.fft

def extract_frequency_spectrum(x: torch.Tensor) -> torch.Tensor:
    """
    Computes the log-magnitude spectrum using 2D FFT.
    Expects x of shape (B, C, H, W).
    """
    # Convert to grayscale if it's RGB
    if x.shape[1] == 3:
        # Standard luminance weights
        weights = torch.tensor([0.299, 0.587, 0.114], device=x.device).view(1, 3, 1, 1)
        x = (x * weights).sum(dim=1, keepdim=True)
        
    # Apply FFT2
    fft_res = torch.fft.fft2(x, norm="ortho")
    # Shift zero frequency to center
    fft_shifted = torch.fft.fftshift(fft_res, dim=(-2, -1))
    
    # Calculate magnitude spectrum and apply log scale for better distribution
    magnitude = torch.abs(fft_shifted)
    log_magnitude = torch.log(magnitude + 1e-8)
    
    # Normalize between 0 and 1 per image
    b, c, h, w = log_magnitude.shape
    log_mag_flat = log_magnitude.view(b, c, -1)
    min_val = log_mag_flat.min(dim=-1, keepdim=True)[0].unsqueeze(-1)
    max_val = log_mag_flat.max(dim=-1, keepdim=True)[0].unsqueeze(-1)
    
    normalized_spectrum = (log_magnitude - min_val) / (max_val - min_val + 1e-8)
    
    return normalized_spectrum
