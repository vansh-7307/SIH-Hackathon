import os
import json
from typing import Dict, List, Optional, Callable
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class SignalScopeDataset(Dataset):
    """
    Core dataset class for SignalScope.
    Loads images and optional metadata, applies transforms, and returns standardized dictionaries.
    """
    
    def __init__(
        self, 
        manifest: List[Dict], 
        transform: Optional[Callable] = None,
        return_metadata: bool = False
    ):
        self.manifest = manifest
        self.transform = transform
        self.return_metadata = return_metadata
        
    def __len__(self) -> int:
        return len(self.manifest)
        
    def __getitem__(self, idx: int) -> Dict:
        item = self.manifest[idx]
        img_path = item['image_path']
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            # Fallback to random noise/black image if corrupted during training to avoid crashing
            # In a real scenario, corrupted files should be filtered out during manifest creation
            image = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
            
        if self.transform:
            image = self.transform(image)
            
        label = item.get('label', -1)
        
        sample = {
            'image': image,
            'label': torch.tensor(label, dtype=torch.float32),
            'generator': item.get('generator', 'unknown')
        }
        
        if self.return_metadata:
            sample['metadata'] = item
            
        return sample

def load_split_manifest(split_path: str) -> Dict[str, List[Dict]]:
    """Loads a saved split manifest."""
    with open(split_path, 'r') as f:
        return json.load(f)

def save_split_manifest(splits: Dict[str, List[Dict]], save_path: str):
    """Saves splits to a JSON manifest."""
    with open(save_path, 'w') as f:
        json.dump(splits, f, indent=4)
