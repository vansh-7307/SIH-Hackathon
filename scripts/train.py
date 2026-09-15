import os
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import csv
import json
import time
from datetime import datetime
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.classifier import SignalScopeModel
from src.data.dataset import SignalScopeDataset
from src.data.transforms import get_train_transforms, get_val_transforms

def build_manifest_from_csv(csv_path, split, base_dir):
    manifest = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['split'] == split:
                cls_str = row['class']
                filename = row['filename']
                img_path = os.path.join(base_dir, split, cls_str, filename)
                manifest.append({
                    'image_path': img_path,
                    'label': int(row['label']),
                    'generator': row['generator'],
                    'generator_family': row['generator_family']
                })
    return manifest

def calc_metrics(y_true, y_probs, y_preds):
    if len(np.unique(y_true)) < 2:
        roc_auc = 0.0
        pr_auc = 0.0
    else:
        roc_auc = roc_auc_score(y_true, y_probs)
        pr_auc = average_precision_score(y_true, y_probs)
        
    f1 = f1_score(y_true, y_preds, average='macro')
    acc = accuracy_score(y_true, y_preds)
    prec = precision_score(y_true, y_preds, zero_division=0)
    rec = recall_score(y_true, y_preds, zero_division=0)
    
    cm = confusion_matrix(y_true, y_preds)
    if cm.shape == (2,2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    else:
        specificity, fpr, tpr = 0, 0, 0
        
    return {
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'macro_f1': f1,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'specificity': specificity,
        'fpr': fpr,
        'tpr': tpr
    }

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_probs = []
    all_preds = []
    all_gens = []
    
    start_time = time.time()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            images = batch['image'].to(device)
            labels = batch['label'].unsqueeze(1).to(device)
            gens = batch['generator']
            
            logits = model(images)
            loss = criterion(logits, labels)
            running_loss += loss.item() * images.size(0)
            
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_gens.extend(gens)
            
    latency = (time.time() - start_time) / max(1, len(dataloader.dataset)) * 1000 # ms per image
    
    y_true = np.array(all_labels).flatten()
    y_probs = np.array(all_probs).flatten()
    y_preds = np.array(all_preds).flatten()
    
    metrics = calc_metrics(y_true, y_probs, y_preds)
    metrics['loss'] = running_loss / len(dataloader.dataset)
    metrics['latency_ms'] = latency
    
    # Per-generator metrics (only meaningful for fake generators)
    gen_metrics = {}
    unique_gens = set(all_gens)
    for g in unique_gens:
        if g.lower() == 'real': continue
        g_indices = [i for i, gen in enumerate(all_gens) if gen == g or y_true[i] == 0]
        if len(g_indices) == 0: continue
        g_true = y_true[g_indices]
        g_probs = y_probs[g_indices]
        g_preds = y_preds[g_indices]
        
        if len(np.unique(g_true)) > 1:
            gen_metrics[g] = roc_auc_score(g_true, g_probs)
            
    metrics['per_generator_auc'] = gen_metrics
    return metrics, y_true, y_probs

def fit_temperature(model, val_loader, device):
    model.eval()
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device)
            labels = batch['label'].unsqueeze(1).to(device)
            logits = model(images)
            all_logits.append(logits)
            all_labels.append(labels)
            
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    
    temperature = nn.Parameter(torch.ones(1, device=device))
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=50)
    
    def eval_lbfgs():
        optimizer.zero_grad()
        loss = criterion(logits / temperature, labels)
        loss.backward()
        return loss
        
    optimizer.step(eval_lbfgs)
    return temperature.item()

def train_one_epoch(model, dataloader, criterion, optimizer, scheduler, device, accum_steps):
    model.train()
    running_loss = 0.0
    all_labels, all_probs, all_preds = [], [], []
    
    optimizer.zero_grad()
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for i, batch in enumerate(pbar):
        images = batch['image'].to(device)
        labels = batch['label'].unsqueeze(1).to(device)
        
        # Simple mixed precision check
        is_cuda = device.type == 'cuda'
        with torch.amp.autocast('cuda') if is_cuda else torch.autocast('cpu', enabled=False):
            logits = model(images)
            loss = criterion(logits, labels)
            loss = loss / accum_steps
            
        loss.backward()
        if (i + 1) % accum_steps == 0 or (i + 1) == len(dataloader):
            optimizer.step()
            optimizer.zero_grad()
            if scheduler: scheduler.step()
            
        running_loss += loss.item() * accum_steps * images.size(0)
        
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        
        all_labels.extend(labels.cpu().detach().numpy())
        all_probs.extend(probs.cpu().detach().numpy())
        all_preds.extend(preds.cpu().detach().numpy())
        
        pbar.set_postfix({'loss': loss.item()*accum_steps})
        
    y_true = np.array(all_labels).flatten()
    y_probs = np.array(all_probs).flatten()
    y_preds = np.array(all_preds).flatten()
    metrics = calc_metrics(y_true, y_probs, y_preds)
    metrics['loss'] = running_loss / len(dataloader.dataset)
    return metrics

def plot_generator_comparison(val_gen, un_gen, out_path):
    plt.figure(figsize=(10, 6))
    gens = list(val_gen.keys()) + list(un_gen.keys())
    aucs = list(val_gen.values()) + list(un_gen.values())
    colors = ['#4f46e5']*len(val_gen) + ['#fbbf24']*len(un_gen)
    
    plt.bar(gens, aucs, color=colors)
    plt.axhline(y=0.5, color='r', linestyle='--', label='Random Guess')
    plt.title('Generator Generalization (ROC-AUC)')
    plt.ylabel('ROC-AUC Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/Tiny-GenImage')
    parser.add_argument('--config', type=str, default='configs/train.yaml')
    parser.add_argument('--experiment', type=str, default='Full SignalScope model', help='Name of experiment')
    parser.add_argument('--baseline', action='store_true', help='Force spatial baseline mode')
    args = parser.parse_args()
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    if args.baseline:
        config['model']['use_frequency'] = False
        config['model']['use_artifact'] = False
        args.experiment = "Spatial baseline"
        
    exp_dir = f"artifacts/models/{args.experiment.replace(' ', '_').lower()}"
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs("artifacts/evaluation", exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    print(f"[{args.experiment}] Using device: {device}")
    
    csv_path = os.path.join(args.data_dir, 'metadata.csv')
    train_manifest = build_manifest_from_csv(csv_path, 'train', args.data_dir)
    val_manifest = build_manifest_from_csv(csv_path, 'validation', args.data_dir)
    unseen_manifest = build_manifest_from_csv(csv_path, 'unseen_generator', args.data_dir)
    
    print(f"Splits -> Train: {len(train_manifest)}, Val: {len(val_manifest)}, Unseen: {len(unseen_manifest)}")
    
    batch_size = 16 if device.type != 'cpu' else 8
    
    train_dataset = SignalScopeDataset(train_manifest, transform=get_train_transforms())
    val_dataset = SignalScopeDataset(val_manifest, transform=get_val_transforms())
    unseen_dataset = SignalScopeDataset(unseen_manifest, transform=get_val_transforms())
    
    num_workers = 0 if sys.platform == 'win32' else 2
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    unseen_loader = DataLoader(unseen_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    model = SignalScopeModel(config.get('model', {})).to(device)
    
    # Overfitting prevention setup
    weight_decay = config.get('weight_decay', 0.01)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.get('learning_rate', 1e-4), weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.get('epochs', 10)*len(train_loader))
    
    epochs = config.get('epochs', 5)
    best_unseen_auc = 0.0
    best_overall_auc = 0.0
    patience_counter = 0
    patience = config.get('early_stopping_patience', 3)
    
    history = []
    
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        train_m = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device, accum_steps=2)
        val_m, _, _ = evaluate(model, val_loader, criterion, device)
        unseen_m, _, _ = evaluate(model, unseen_loader, criterion, device)
        
        print(f"Train     -> Loss: {train_m['loss']:.4f} | AUC: {train_m['roc_auc']:.4f} | Acc: {train_m['accuracy']:.4f}")
        print(f"Val       -> Loss: {val_m['loss']:.4f} | AUC: {val_m['roc_auc']:.4f} | Acc: {val_m['accuracy']:.4f}")
        print(f"UnseenGen -> Loss: {unseen_m['loss']:.4f} | AUC: {unseen_m['roc_auc']:.4f} | Acc: {unseen_m['accuracy']:.4f}")
        
        history.append({'epoch': epoch, 'train': train_m, 'val': val_m, 'unseen': unseen_m})
        
        # Model selection logic primarily based on Unseen-Generator AUC
        if unseen_m['roc_auc'] > best_unseen_auc or (unseen_m['roc_auc'] == best_unseen_auc and val_m['roc_auc'] > best_overall_auc):
            best_unseen_auc = unseen_m['roc_auc']
            best_overall_auc = val_m['roc_auc']
            patience_counter = 0
            
            torch.save(model.state_dict(), os.path.join(exp_dir, 'best_model.pt'))
            print(">> Saved new best checkpoint!")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break
                
    # Load best model for calibration
    model.load_state_dict(torch.load(os.path.join(exp_dir, 'best_model.pt'), weights_only=True))
    print("\nFitting Temperature Scaling Calibration...")
    temp = fit_temperature(model, val_loader, device)
    print(f"Optimal Temperature: {temp:.4f}")
    
    # Save config with temp
    config['calibration'] = {'temperature': temp}
    with open(os.path.join(exp_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
        
    # Final evaluation
    val_m, _, _ = evaluate(model, val_loader, criterion, device)
    unseen_m, _, _ = evaluate(model, unseen_loader, criterion, device)
    
    final_metrics = {
        'val_metrics': val_m,
        'unseen_metrics': unseen_m,
        'temperature': temp,
        'hardware': str(device)
    }
    
    with open(os.path.join(exp_dir, 'metrics.json'), 'w') as f:
        json.dump(final_metrics, f, indent=4)
        
    with open(os.path.join(exp_dir, 'training_summary.json'), 'w') as f:
        # Avoid non-serializable elements in history
        cleaned_history = []
        for h in history:
            ch = {'epoch': h['epoch']}
            for split in ['train', 'val', 'unseen']:
                ch[split] = {k: v for k, v in h[split].items() if k != 'per_generator_auc'}
            cleaned_history.append(ch)
        json.dump(cleaned_history, f, indent=4)
        
    with open('artifacts/evaluation/unseen_generator_results.json', 'w') as f:
        json.dump(final_metrics, f, indent=4)
        
    plot_generator_comparison(val_m.get('per_generator_auc', {}), unseen_m.get('per_generator_auc', {}), 'artifacts/evaluation/generator_comparison.png')
    print("Metrics and plots saved.")

    # Expose the best model globally
    global_model_path = 'artifacts/models/signalscope_best.pt' if not args.baseline else 'artifacts/models/baseline_best.pt'
    torch.save(model.state_dict(), global_model_path)
    print(f"Global checkpoint exposed to {global_model_path}")
    
    # Optional config update for web API
    api_config_path = 'configs/inference.yaml'
    if os.path.exists(api_config_path):
        with open(api_config_path, 'r') as f:
            api_cfg = yaml.safe_load(f)
        api_cfg['model_path'] = global_model_path
        api_cfg['calibration_temp'] = temp
        api_cfg['model'] = config['model'] # ensure API uses correct architecture
        with open(api_config_path, 'w') as f:
            yaml.dump(api_cfg, f)
            
if __name__ == '__main__':
    main()
