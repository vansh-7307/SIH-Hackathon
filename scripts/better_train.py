import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import random

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.classifier import SignalScopeModel
from src.data.dataset import SignalScopeDataset
from src.data.leakage_detection import create_manifest
from src.data.transforms import get_train_transforms

def better_train():
    torch.set_num_threads(16)
    print("Starting Enhanced Training (Balanced Subset of 2500 images, 3 Epochs)...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    with open('configs/train.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    manifest = create_manifest("D:\\vivek\\train")
    
    # Ensure perfectly balanced dataset for the fast run
    real_imgs = [m for m in manifest if m['label'] == 0]
    fake_imgs = [m for m in manifest if m['label'] == 1]
    random.shuffle(real_imgs)
    random.shuffle(fake_imgs)
    
    manifest = real_imgs[:1250] + fake_imgs[:1250]
    random.shuffle(manifest)
    
    dataset = SignalScopeDataset(manifest, transform=get_train_transforms())
    # Use batch_size 32 for faster CPU throughput on 16 cores
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = SignalScopeModel(config.get('model', {})).to(device)
    
    # Keep heavy backbone frozen, but train classifiers aggressively
    model.spatial.freeze()
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3)
    
    model.train()
    for epoch in range(3):
        correct = 0
        total = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/3")
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
            
            pbar.set_postfix({'acc': f'{correct/total:.3f}'})
        scheduler.step()
            
    os.makedirs('model/weights', exist_ok=True)
    torch.save(model.state_dict(), 'model/weights/best_model.pt')
    print("Enhanced training complete! Saved to best_model.pt")

if __name__ == '__main__':
    better_train()
