import os
import json
import random
import argparse
from pathlib import Path
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('--jpg_root',   type=str, default='/kaggle/working/FYP_DATA_jpg_raw')
parser.add_argument('--output',     type=str, default='/kaggle/working/FYP_DATA_jpg_raw/dataset.json')
parser.add_argument('--val_split',  type=float, default=0.2)
parser.add_argument('--seed',       type=int, default=42)
args = parser.parse_args()

# Match the classes used in your training/testing
CLASSES = ['fight', 'unsafeClimb', 'unsafeJump', 'unsafeThrow', 'fall']
random.seed(args.seed)

# This dictionary will group augmented videos by their original source
# Format: { class_name: { base_video_id: [list_of_folders] } }
video_groups = {c: defaultdict(list) for c in CLASSES}

print(f"Scanning {args.jpg_root}...")

for cls in CLASSES:
    cls_dir = os.path.join(args.jpg_root, cls)
    if not os.path.exists(cls_dir):
        print(f"  [WARN] Missing folder: {cls}")
        continue

    for vid_name in os.listdir(cls_dir):
        vid_dir = os.path.join(cls_dir, vid_name)
        if not os.path.isdir(vid_dir): continue
        
        frames = [f for f in os.listdir(vid_dir) if f.endswith('.jpg')]
        if len(frames) == 0: continue

        # --- KEY FIX: Identify the Base Video ID ---
        # If vid_name is 'aug_RandomRotate_fi012', base_id becomes 'fi012'
        # If vid_name is 'fi012', base_id stays 'fi012'
        base_id = vid_name.split('_')[-1] 
        
        video_groups[cls][base_id].append((vid_name, len(frames)))

database = {}
print("\nSplitting data (Grouping by original video source to prevent leakage)...")
print(f"{'Class':15s} {'Unique Vids':>12s} {'Total Folders':>14s} {'Train':>8s} {'Val':>8s}")
print("-" * 65)

for cls in CLASSES:
    base_ids = list(video_groups[cls].keys())
    random.shuffle(base_ids) # Shuffle original videos, not individual folders

    n_unique = len(base_ids)
    n_val_unique = max(1, int(n_unique * args.val_split))
    
    val_base_ids = set(base_ids[:n_val_unique])
    
    cls_train_count = 0
    cls_val_count = 0
    total_cls_folders = 0

    for base_id, folders in video_groups[cls].items():
        subset = 'validation' if base_id in val_base_ids else 'training'
        
        for vid_name, n_frames in folders:
            database[vid_name] = {
                'subset': subset,
                'annotations': {'label': cls, 'segment': [1, n_frames]}
            }
            if subset == 'training': cls_train_count += 1
            else: cls_val_count += 1
            total_cls_folders += 1

    print(f"{cls:15s} {n_unique:12d} {total_cls_folders:14d} {cls_train_count:8d} {cls_val_count:8d}")

# Save the JSON
os.makedirs(os.path.dirname(args.output), exist_ok=True)
with open(args.output, 'w') as f:
    json.dump({'labels': CLASSES, 'database': database}, f, indent=2)

print(f"\n✅ Grouped Annotation saved to: {args.output}")
