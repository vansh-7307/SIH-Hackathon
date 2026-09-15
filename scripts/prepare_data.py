import os
import argparse
import json
import hashlib
import csv
from datasets import load_dataset
from PIL import Image
import imagehash
from tqdm import tqdm
import random

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--download', action='store_true', help='Download dataset')
    parser.add_argument('--limit', type=int, default=2000, help='Total images limit')
    return parser.parse_args()

def compute_hashes(img):
    img_bytes = img.tobytes()
    sha256 = hashlib.sha256(img_bytes).hexdigest()
    phash = str(imagehash.phash(img))
    return sha256, phash

def main():
    args = parse_args()
    
    base_dir = "data/Tiny-GenImage"
    splits = ['train', 'validation', 'unseen_generator']
    classes = ['real', 'fake']
    
    for split in splits:
        for cls in classes:
            os.makedirs(os.path.join(base_dir, split, cls), exist_ok=True)
            
    print("Downloading Tiny-GenImage dataset...")
    ds = load_dataset('TheKernel01/Tiny-GenImage', split='train')
    
    target_total = args.limit
    target_real = target_total // 2
    target_fake = target_total - target_real
    
    generator_names = ['Real', 'ADM', 'BigGAN', 'GLIDE', 'Midjourney', 'SD14', 'SD15', 'VQDM', 'Wukong']
    fake_generators = generator_names[1:]
    
    unseen_gen = 'Wukong'
    seen_gens = [g for g in fake_generators if g != unseen_gen]
    
    # Target allocations
    # Fake: ~125 per generator. Unseen entirely in unseen_generator
    # Seen: 100 train, 25 val per generator
    target_fake_per_gen = target_fake // len(fake_generators)
    
    counts = {
        'real': {'train': 0, 'validation': 0, 'unseen_generator': 0},
        'fake': {g: {'train': 0, 'validation': 0, 'unseen_generator': 0} for g in fake_generators}
    }
    
    collected_sha = set()
    collected_phash = set()
    metadata = []
    
    corrupt_count = 0
    dup_sha_count = 0
    dup_phash_count = 0
    
    print(f"Collecting {target_total} images with {unseen_gen} as the unseen generator...")
    
    for item in ds:
        # Check limits
        total_real = sum(counts['real'].values())
        total_fake = sum(sum(gen_counts.values()) for gen_counts in counts['fake'].values())
        
        if total_real >= target_real and total_fake >= target_fake:
            break
            
        try:
            img = item['image'].convert('RGB')
        except Exception:
            corrupt_count += 1
            continue
            
        sha, phash = compute_hashes(img)
        if sha in collected_sha:
            dup_sha_count += 1
            continue
        if phash in collected_phash:
            dup_phash_count += 1
            continue
            
        label_int = item['label'] # 0: real, 1: fake
        label_str = 'real' if label_int == 0 else 'fake'
        gen_str = generator_names[item['generator']]
        
        # Decide split
        target_split = None
        
        if label_str == 'real':
            if total_real < target_real:
                # 70% train, 15% val, 15% unseen
                if counts['real']['train'] < target_real * 0.7:
                    target_split = 'train'
                elif counts['real']['validation'] < target_real * 0.15:
                    target_split = 'validation'
                else:
                    target_split = 'unseen_generator'
        else:
            if total_fake < target_fake:
                gen_total = sum(counts['fake'][gen_str].values())
                if gen_str == unseen_gen:
                    target_split = 'unseen_generator'
                else:
                    # 80% train, 20% val
                    if counts['fake'][gen_str]['train'] < max(target_fake_per_gen * 0.8, gen_total * 0.8 + 1):
                        target_split = 'train'
                    else:
                        target_split = 'validation'
        
        if target_split is None:
            continue
            
        # Save image
        idx = total_real + total_fake
        filename = f"{label_str}_{gen_str}_{idx:05d}.jpg".lower()
        filepath = os.path.join(base_dir, target_split, label_str, filename)
        
        img.save(filepath, 'JPEG', quality=95)
        
        # Update trackers
        collected_sha.add(sha)
        collected_phash.add(phash)
        
        if label_str == 'real':
            counts['real'][target_split] += 1
        else:
            counts['fake'][gen_str][target_split] += 1
            
        metadata.append({
            'filename': filename,
            'label': label_int,
            'class': label_str,
            'generator': gen_str,
            'generator_family': 'diffusion' if gen_str in ['ADM', 'GLIDE', 'Midjourney', 'SD14', 'SD15', 'Wukong'] else 'gan',
            'dataset_source': 'Tiny-GenImage',
            'split': target_split,
            'width': img.width,
            'height': img.height,
            'format': 'JPEG',
            'sha256': sha,
            'perceptual_hash': phash
        })
        
        total_collected = sum(counts['real'].values()) + sum(sum(gen_counts.values()) for gen_counts in counts['fake'].values())
        if total_collected % 100 == 0:
            print(f"Collected {total_collected} / {target_total} images")
            
    # Write metadata
    print("Writing metadata.csv...")
    with open(os.path.join(base_dir, 'metadata.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=metadata[0].keys())
        writer.writeheader()
        writer.writerows(metadata)
        
    # Write report
    report = {
        'total_images': len(metadata),
        'real_count': sum(counts['real'].values()),
        'fake_count': sum(sum(gen_counts.values()) for gen_counts in counts['fake'].values()),
        'generator_counts': counts['fake'],
        'generator_families': ['diffusion', 'gan'],
        'corrupt_images_removed': corrupt_count,
        'duplicates_removed': dup_sha_count,
        'near_duplicates_removed': dup_phash_count,
        'train_count': sum(1 for m in metadata if m['split'] == 'train'),
        'validation_count': sum(1 for m in metadata if m['split'] == 'validation'),
        'unseen_generator_count': sum(1 for m in metadata if m['split'] == 'unseen_generator')
    }
    
    with open(os.path.join(base_dir, 'dataset_report.json'), 'w') as f:
        json.dump(report, f, indent=4)
        
    md_content = f"""# Tiny-GenImage Dataset Report

## Summary
- **Total Images:** {report['total_images']}
- **Real:** {report['real_count']}
- **Fake:** {report['fake_count']}

## Splits
- **Train:** {report['train_count']}
- **Validation:** {report['validation_count']}
- **Unseen Generator Validation:** {report['unseen_generator_count']}

## Fake Generator Breakdown
"""
    for gen, splits in counts['fake'].items():
        md_content += f"- **{gen}:** Train({splits['train']}), Val({splits['validation']}), Unseen({splits['unseen_generator']})\n"
        
    with open(os.path.join(base_dir, 'dataset_report.md'), 'w') as f:
        f.write(md_content)
        
    print("Data preparation complete! Report saved.")

if __name__ == '__main__':
    main()
