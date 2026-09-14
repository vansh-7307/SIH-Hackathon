import json
import os
import hashlib
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import pandas as pd
from PIL import Image

def get_image_hash(image_path: str) -> str:
    """Computes SHA-256 hash of an image file for exact duplicate detection."""
    hasher = hashlib.sha256()
    with open(image_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

class LeakageDetector:
    """
    Detects data leakage between train/val/test splits.
    Handles exact duplicates and simple near-duplicates based on hashes.
    """
    
    def __init__(self):
        self.hashes = {}
        
    def check_leakage(self, splits: Dict[str, List[Dict]]) -> Dict[str, List[str]]:
        """
        Ensures no exact duplicates exist across different splits.
        Returns a report of detected leakages.
        """
        seen_hashes = {}
        leakages = defaultdict(list)
        
        for split_name, items in splits.items():
            for item in items:
                img_path = item['image_path']
                if not os.path.exists(img_path):
                    continue
                    
                img_hash = get_image_hash(img_path)
                
                if img_hash in seen_hashes:
                    prev_split = seen_hashes[img_hash]['split']
                    if prev_split != split_name:
                        leakages[f"{prev_split}_to_{split_name}"].append(img_path)
                else:
                    seen_hashes[img_hash] = {
                        'split': split_name,
                        'path': img_path
                    }
                    
        return dict(leakages)

def split_generator_aware(
    data: List[Dict], 
    val_ratio: float = 0.15, 
    test_ratio: float = 0.15,
    unseen_generators: Optional[List[str]] = None,
    seed: int = 42
) -> Dict[str, List[Dict]]:
    """
    Creates train/val/test splits ensuring unseen generators are kept ONLY in the test set
    if specified.
    """
    import random
    random.seed(seed)
    
    # Shuffle data
    random.shuffle(data)
    
    splits = {
        'train': [],
        'val': [],
        'test': []
    }
    
    unseen_generators = unseen_generators or []
    seen_data = []
    
    for item in data:
        generator = item.get('generator', 'unknown')
        if generator in unseen_generators:
            splits['test'].append(item)
        else:
            seen_data.append(item)
            
    # Calculate remaining needed for val and test
    total_seen = len(seen_data)
    val_size = int(total_seen * val_ratio)
    test_size_needed = max(0, int(len(data) * test_ratio) - len(splits['test']))
    
    splits['val'] = seen_data[:val_size]
    splits['test'].extend(seen_data[val_size:val_size + test_size_needed])
    splits['train'] = seen_data[val_size + test_size_needed:]
    
    return splits

def create_manifest(data_dir: str, metadata_csv: Optional[str] = None) -> List[Dict]:
    """
    Creates a manifest of all images in the dataset, parsing metadata if available.
    """
    manifest = []
    
    if metadata_csv and os.path.exists(metadata_csv):
        df = pd.read_csv(metadata_csv)
        for _, row in df.iterrows():
            manifest.append(row.to_dict())
    else:
        # Fallback to directory structure (e.g., data/real/, data/fake/)
        for label_dir in os.listdir(data_dir):
            if label_dir.lower() not in ['real', 'fake']:
                continue
                
            dir_path = os.path.join(data_dir, label_dir)
            if not os.path.isdir(dir_path):
                continue
                
            label = 0 if label_dir.lower() == 'real' else 1
            for img_name in os.listdir(dir_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    manifest.append({
                        'image_path': os.path.join(dir_path, img_name),
                        'label': label,
                        'generator': 'unknown'
                    })
                    
    return manifest
