import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import load_dataset
from torchvision import transforms
from PIL import Image

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.classifier import SignalScopeModel

def train_genimage_stream():
    """
    Streams the massive GenImage dataset directly from HuggingFace Cloud 
    without needing to download the 100GB files to your local hard drive.
    """
    print("================================================================")
    print(" CONNECTING TO GENIMAGE CLOUD DATABASE (MILLION-SCALE DATASET) ")
    print("================================================================")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model config
    with open('configs/train.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    model = SignalScopeModel(config.get('model', {})).to(device)
    model.spatial.freeze()
    
    # Try to connect to GenImage on HuggingFace Hub in Streaming Mode
    try:
        print("\nEstablishing secure streaming connection to 'andrew-zhu/genimage-dataset'...")
        # We use streaming=True so it doesn't download hundreds of GBs!
        dataset = load_dataset("andrew-zhu/genimage-dataset", streaming=True, split="train")
        print("Connection successful! Streaming data directly into neural network.")
    except Exception as e:
        print(f"Cloud connection failed: {e}")
        print("Please ensure you have an active internet connection.")
        return

    # Basic transforms for the incoming stream
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    model.train()
    print("\nStarting cloud-streamed training (Processing first 5000 images for demo)...")
    
    correct = 0
    total = 0
    
    # Manually iterate over the stream
    for idx, item in enumerate(dataset):
        if idx >= 5000:
            break
            
        try:
            # HuggingFace datasets usually return a PIL image
            image = item['image'].convert('RGB')
            # 0 for real, 1 for fake
            label = item['label']
            
            x = transform(image).unsqueeze(0).to(device)
            y = torch.tensor([[label]], dtype=torch.float32).to(device)
            
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            correct += (preds == y).sum().item()
            total += 1
            
            if total % 100 == 0:
                print(f"Streamed {total}/5000 images | Live Accuracy: {correct/total:.3f} | Loss: {loss.item():.4f}")
                
        except Exception as e:
            continue

    print("\n================================================================")
    print(" CLOUD TRAINING COMPLETE ")
    print("================================================================")
    os.makedirs('model/weights', exist_ok=True)
    torch.save(model.state_dict(), 'model/weights/best_model.pt')
    print("GenImage-trained intelligence saved to best_model.pt")

if __name__ == '__main__':
    train_genimage_stream()
