import os
import argparse
import csv
import json
from datasets import load_dataset
from PIL import Image
import imagehash
import hashlib

def compute_hashes(img):
    sha = hashlib.sha256(img.tobytes()).hexdigest()
    phash = str(imagehash.phash(img))
    return sha, phash

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/Tiny-GenImage')
    parser.add_argument('--target_real', type=int, default=1500)
    parser.add_argument('--target_fake', type=int, default=1500)
    args = parser.parse_args()

    base_dir = args.data_dir
    csv_path = os.path.join(base_dir, 'metadata.csv')
    report_path = os.path.join(base_dir, 'dataset_report.json')
    
    # Load existing metadata
    existing_metadata = []
    collected_sha = set()
    collected_phash = set()
    counts = {
        'real': {'train': 0, 'validation': 0, 'unseen_generator': 0},
        'fake': {}
    }
    
    generator_names = ['Real', 'ADM', 'BigGAN', 'GLIDE', 'Midjourney', 'SD14', 'SD15', 'VQDM', 'Wukong']
    
    print("Loading existing dataset metadata...")
    if os.path.exists(csv_path):
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_metadata.append(row)
                collected_sha.add(row['sha256'])
                collected_phash.add(row['perceptual_hash'])
                
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
    
    print(f"Current dataset: {total_real} REAL, {total_fake} FAKE")
    
    if total_real >= args.target_real and total_fake >= args.target_fake:
        print("Target already reached.")
        return
        
    print("Downloading/loading Tiny-GenImage dataset to find new images...")
    ds = load_dataset('TheKernel01/Tiny-GenImage', split='train')
    
    # Identify unseen generator (usually Wukong)
    unseen_gen = 'Wukong'
    
    new_metadata = []
    rejections = []
    
    total_scanned = 0
    
    # Helper to calculate generator distribution so we can balance
    def get_fake_gen_count(g_name):
        return sum(counts['fake'].get(g_name, {'train': 0}).values())
    
    for item in ds:
        if total_real >= args.target_real and total_fake >= args.target_fake:
            break
            
        total_scanned += 1
        
        try:
            img = item['image'].convert('RGB')
        except Exception as e:
            rejections.append({'reason': 'corrupt', 'index': total_scanned})
            continue
            
        if img.width < 128 or img.height < 128:
            rejections.append({'reason': 'low_res', 'index': total_scanned})
            continue
            
        sha, phash = compute_hashes(img)
        if sha in collected_sha or phash in collected_phash:
            continue # Silent rejection for dupes since it's normal
            
        label_int = item['label']
        label_str = 'real' if label_int == 0 else 'fake'
        gen_str = generator_names[item['generator']]
        
        target_split = None
        
        if label_str == 'real' and total_real < args.target_real:
            # 70% train, 15% val, 15% unseen
            if counts['real']['train'] < args.target_real * 0.7:
                target_split = 'train'
            elif counts['real']['validation'] < args.target_real * 0.15:
                target_split = 'validation'
            else:
                target_split = 'unseen_generator'
        elif label_str == 'fake' and total_fake < args.target_fake:
            if gen_str == unseen_gen:
                target_split = 'unseen_generator'
            else:
                # Prefer underrepresented generators
                # We want to balance train/val as 80/20 per generator roughly
                if gen_str not in counts['fake']:
                    counts['fake'][gen_str] = {'train': 0, 'validation': 0, 'unseen_generator': 0}
                    
                g_total = get_fake_gen_count(gen_str)
                # If we have 7 seen generators, ideal is (target_fake - unseen_count) / 7.
                # Just add it if we need fakes.
                
                # Check if we should aggressively skip overrepresented ones
                # Average per generator (if 7 seen generators) should be roughly 200.
                if g_total > 250 and total_scanned < 20000:
                    # Skip for now to let other generators catch up, unless we're running out of images
                    continue
                
                # 80/20 train/val
                if counts['fake'][gen_str]['train'] <= g_total * 0.8:
                    target_split = 'train'
                else:
                    target_split = 'validation'
                    
        if target_split is None:
            continue
            
        # Save image
        idx = total_real + total_fake + 1
        filename = f"{label_str}_{gen_str}_{idx:05d}.jpg".lower()
        filepath = os.path.join(base_dir, target_split, label_str, filename)
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        img.save(filepath, 'JPEG', quality=95)
        
        collected_sha.add(sha)
        collected_phash.add(phash)
        
        if label_str == 'real':
            counts['real'][target_split] += 1
            total_real += 1
        else:
            counts['fake'][gen_str][target_split] += 1
            total_fake += 1
            
        new_row = {
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
        }
        new_metadata.append(new_row)
        existing_metadata.append(new_row)
        
        if (total_real + total_fake) % 100 == 0:
            print(f"Collected {total_real} Real, {total_fake} Fake")
            
    print("Writing updated metadata.csv...")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=existing_metadata[0].keys())
        writer.writeheader()
        writer.writerows(existing_metadata)
        
    os.makedirs('artifacts/data', exist_ok=True)
    with open('artifacts/data/additional_1000_rejections.json', 'w') as f:
        json.dump(rejections, f, indent=4)
        
    print(f"Done! Final Dataset: {total_real} Real, {total_fake} Fake")
    
if __name__ == '__main__':
    main()
