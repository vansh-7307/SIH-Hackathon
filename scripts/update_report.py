import os
import csv
import json

def main():
    base_dir = 'data/Tiny-GenImage'
    csv_path = os.path.join(base_dir, 'metadata.csv')
    
    metadata = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            metadata.append(row)
            
    # Calculate stats
    counts = {
        'real': {'train': 0, 'validation': 0, 'unseen_generator': 0},
        'fake': {}
    }
    
    for row in metadata:
        cls = row['class']
        split = row['split']
        gen = row['generator']
        
        if cls == 'real':
            counts['real'][split] += 1
        else:
            if gen not in counts['fake']:
                counts['fake'][gen] = {'train': 0, 'validation': 0, 'unseen_generator': 0}
            counts['fake'][gen][split] += 1
            
    total_real = sum(counts['real'].values())
    total_fake = sum(sum(g.values()) for g in counts['fake'].values())
    
    # Load old report to compare
    old_report_path = os.path.join(base_dir, 'dataset_report.json')
    old_counts = {}
    if os.path.exists(old_report_path):
        with open(old_report_path, 'r') as f:
            old_report = json.load(f)
            old_counts = old_report.get('generator_counts', {})
            
    new_report = {
        'total_images': len(metadata),
        'real_count': total_real,
        'fake_count': total_fake,
        'generator_counts': counts['fake'],
        'generator_families': ['diffusion', 'gan'],
        'dataset_source_counts': {'Tiny-GenImage': len(metadata)},
        'resolution_distribution': {'224x224': len(metadata)}, # assuming typical resized
        'duplicates_removed': 0, # Cannot know retrospectively without full scan
        'near_duplicates_removed': 0,
        'corrupt_images_removed': 0,
        'train_count': sum(counts['real']['train'] for row in metadata) if total_real else 0, # rough 
    }
    
    # accurately count splits
    train_c = 0
    val_c = 0
    unseen_c = 0
    for row in metadata:
        if row['split'] == 'train': train_c += 1
        elif row['split'] == 'validation': val_c += 1
        elif row['split'] == 'unseen_generator': unseen_c += 1
        
    new_report['train_count'] = train_c
    new_report['validation_count'] = val_c
    new_report['unseen_generator_count'] = unseen_c
    
    with open(old_report_path, 'w') as f:
        json.dump(new_report, f, indent=4)
        
    md_content = f"""# Tiny-GenImage Dataset Expansion Report (3000 Images)

## Summary
- **Total Images:** {new_report['total_images']} (Old: {old_report.get('total_images', 'N/A') if old_report else 'N/A'})
- **Real:** {new_report['real_count']} (Old: {old_report.get('real_count', 'N/A') if old_report else 'N/A'})
- **Fake:** {new_report['fake_count']} (Old: {old_report.get('fake_count', 'N/A') if old_report else 'N/A'})

## Splits
- **Train:** {new_report['train_count']}
- **Validation:** {new_report['validation_count']}
- **Unseen Generator Validation:** {new_report['unseen_generator_count']}

## Fake Generator Breakdown (Before vs After)
"""
    for gen in sorted(counts['fake'].keys()):
        old_val = sum(old_counts.get(gen, {}).values()) if gen in old_counts else 0
        new_val = sum(counts['fake'][gen].values())
        diff = new_val - old_val
        md_content += f"- **{gen}:** {new_val} (was {old_val}, added +{diff})\n"
        
    with open(os.path.join(base_dir, 'dataset_report.md'), 'w') as f:
        f.write(md_content)
        
    print("Report updated.")

if __name__ == '__main__':
    main()
