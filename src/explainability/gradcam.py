import torch
import torch.nn.functional as F
import numpy as np
import cv2

class GradCAM:
    """
    Produces class activation maps to visualize spatial feature importance.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        self.activations = output
        
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
        
    def __call__(self, x, class_idx=None):
        self.model.eval()
        
        # Forward pass
        logits, features = self.model(x, return_features=True)
        
        if class_idx is None:
            # For binary classification, use the single output logit
            score = logits[:, 0]
        else:
            score = logits[:, class_idx]
            
        self.model.zero_grad()
        score.backward(retain_graph=True)
        
        # Compute weights
        gradients = self.gradients.detach().cpu().numpy()
        activations = self.activations.detach().cpu().numpy()
        
        b, k, u, v = gradients.shape
        alpha = np.mean(gradients, axis=(2, 3), keepdims=True)
        
        # Weighted combination of activations
        cam = np.sum(alpha * activations, axis=1, keepdims=True)
        cam = np.maximum(cam, 0) # ReLU
        
        # Normalize
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-8)
        
        return cam[0, 0] # Return 2D array for the first image in batch

def generate_overlay(image_np: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Overlays heatmap on original image."""
    heatmap_resized = cv2.resize(heatmap, (image_np.shape[1], image_np.shape[0]))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    
    # Convert BGR to RGB
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    
    # Overlay
    overlay = cv2.addWeighted(image_np, 1 - alpha, heatmap_color, alpha, 0)
    return overlay
