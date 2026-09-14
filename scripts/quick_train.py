import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import random

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.classifier import SignalScopeModel
from src.data.dataset import SignalScopeDataset
from src.data.leakage_detection import create_manifest
from src.data.transforms import get_train_transforms

def fast_train():
    print("Starting Fast-Track Training (Subset of 1000 images)...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    with open('configs/train.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    manifest = create_manifest("D:\\vivek\\train")
    # Take a random subset of 1000 images for speed
    random.shuffle(manifest)
    manifest = manifest[:1000]
    
    dataset = SignalScopeDataset(manifest, transform=get_train_transforms())
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    model = SignalScopeModel(config.get('model', {})).to(device)
    
    # Freeze the heavy spatial backbone for super fast training
    model.spatial.freeze()
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(1): # Just 1 epoch for immediate results
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(dataloader, desc="Fast Training")
        for batch in pbar:
            images = batch['image'].to(device)
            labels = batch['label'].unsqueeze(1).to(device)
            
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            pbar.set_postfix({'acc': correct/total})
            
    os.makedirs('model/weights', exist_ok=True)
    torch.save(model.state_dict(), 'model/weights/best_model.pt')
    print("Fast training complete! Saved to best_model.pt")

if __name__ == '__main__':
    fast_train()
