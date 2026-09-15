import json
import os
import yaml
import shutil

def main():
    old_metrics_path = 'artifacts/models/full_signalscope_model/metrics.json'
    new_metrics_path = 'artifacts/models/full_signalscope_model_3k/metrics.json'
    
    with open(old_metrics_path, 'r') as f:
        old_metrics = json.load(f)
        
    with open(new_metrics_path, 'r') as f:
        new_metrics = json.load(f)
        
    print("\n============================================================")
    print("EXPERIMENT COMPARISON")
    print("============================================================")
    
    print(f"{'Metric':<25} | {'2K MODEL (Old)':<15} | {'3K MODEL (New)':<15}")
    print("-" * 60)
    
    def print_metric(name, key, source='val_metrics'):
        old_val = old_metrics[source].get(key, 0)
        new_val = new_metrics[source].get(key, 0)
        print(f"{name:<25} | {old_val:<15.4f} | {new_val:<15.4f}")
        
    print_metric('Validation AUC', 'roc_auc', 'val_metrics')
    print_metric('Unseen AUC', 'roc_auc', 'unseen_metrics')
    print_metric('Macro-F1', 'macro_f1', 'val_metrics')
    print_metric('Accuracy', 'accuracy', 'val_metrics')
    print_metric('FPR', 'fpr', 'val_metrics')
    
    # Model Selection Logic
    old_unseen = old_metrics['unseen_metrics'].get('roc_auc', 0)
    new_unseen = new_metrics['unseen_metrics'].get('roc_auc', 0)
    old_val = old_metrics['val_metrics'].get('roc_auc', 0)
    new_val = new_metrics['val_metrics'].get('roc_auc', 0)
    
    print("-" * 60)
    
    promote = False
    if new_unseen > old_unseen or (new_unseen == old_unseen and new_val > old_val):
        promote = True
        
    if promote:
        print("DECISION: Promoting 3K Model! It showed better generalization.")
        best_model_path = 'artifacts/models/full_signalscope_model_3k/best_model.pt'
        temp = new_metrics['temperature']
    else:
        print("DECISION: Keeping 2K Model. The 3K model did not improve generalization.")
        best_model_path = 'artifacts/models/full_signalscope_model/best_model.pt'
        temp = old_metrics['temperature']
        
    # Promote checkpoint
    shutil.copy(best_model_path, 'artifacts/models/signalscope_best.pt')
    
    # Update configs/inference.yaml
    with open('configs/inference.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
        
    cfg['model_path'] = 'artifacts/models/signalscope_best.pt'
    cfg['calibration_temp'] = temp
    cfg['dataset_version'] = '3000_images' if promote else '2000_images'
    cfg['real_count'] = 1500 if promote else 1000
    cfg['fake_count'] = 1500 if promote else 1000
    
    with open('configs/inference.yaml', 'w') as f:
        yaml.dump(cfg, f)
        
    print(f"Updated configs/inference.yaml with dataset_version: {cfg['dataset_version']}")

if __name__ == '__main__':
    main()
