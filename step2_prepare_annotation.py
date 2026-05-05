"""
STEP 2: prepare_annotation.py
Generates dataset.json from JPG frames folder.
UCF101 format compatible with SlowFast training script.

Output JSON structure:
{
  "labels": [...],
  "database": {
    "video_name": {
      "subset": "training" | "validation" | "testing",
      "annotations": {
        "label": "fight",
        "segment": [1, N]
      }
    }
  }
}

Usage:
    python prepare_annotation.py \
        --jpg_root "G:/My Drive/Segmented_FYP_DATA_jpg_raw" \
        --output   "G:/My Drive/Segmented_FYP_DATA_jpg_raw/dataset.json" \
        --val_split 0.2 \
        --test_split 0.15
"""

import os
import json
import random
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--jpg_root',   type=str,
                    default='G:/My Drive/Segmented_FYP_DATA_jpg_raw')
parser.add_argument('--output',     type=str,
                    default='G:/My Drive/Segmented_FYP_DATA_jpg_raw/dataset.json')
parser.add_argument('--val_split',  type=float, default=0.2,
                    help='Fraction for validation')
parser.add_argument('--test_split', type=float, default=0.15,
                    help='Fraction for testing')
parser.add_argument('--seed',       type=int, default=42)
args = parser.parse_args()

CLASSES = ['fight', 'Normal', 'unsafeClimb', 'unsafeJump', 'unsafeThrow']
random.seed(args.seed)

database = {}
per_class = {c: [] for c in CLASSES}
missing   = []

for cls in CLASSES:
    cls_dir = os.path.join(args.jpg_root, cls)
    if not os.path.exists(cls_dir):
        missing.append(cls)
        continue

    for vid_name in os.listdir(cls_dir):
        vid_dir = os.path.join(cls_dir, vid_name)
        if not os.path.isdir(vid_dir):
            continue
        frames = [f for f in os.listdir(vid_dir) if f.endswith('.jpg')]
        if len(frames) == 0:
            print(f"  [SKIP] No frames: {vid_dir}")
            continue
        per_class[cls].append((vid_name, len(frames)))

# print class counts
print("\nClass distribution:")
total_videos = 0
for cls in CLASSES:
    n = len(per_class[cls])
    total_videos += n
    print(f"  {cls:15s}: {n} videos")
print(f"  {'TOTAL':15s}: {total_videos} videos")
if missing:
    print(f"\n[WARN] Missing class folders: {missing}")

# print counts first, then split
print("\nDetailed count before splitting:")
print(f"{'Class':15s} {'Total':>8s} {'Train':>8s} {'Val':>8s} {'Test':>8s}")
print("-" * 50)

split_counts = {'training': 0, 'validation': 0, 'testing': 0}

for cls in CLASSES:
    vids = per_class[cls]
    random.shuffle(vids)

    n       = len(vids)
    n_val   = max(1, int(n * args.val_split))
    n_test  = max(1, int(n * args.test_split))
    n_train = n - n_val - n_test

    print(f"{cls:15s} {n:>8d} {n_train:>8d} {n_val:>8d} {n_test:>8d}")

    splits = (
        [('training',   v) for v in vids[:n_train]] +
        [('validation', v) for v in vids[n_train:n_train+n_val]] +
        [('testing',    v) for v in vids[n_train+n_val:]]
    )

    for subset, (vid_name, n_frames) in splits:
        database[vid_name] = {
            'subset': subset,
            'annotations': {
                'label': cls,
                'segment': [1, n_frames]
            }
        }
        split_counts[subset] += 1

print("-" * 50)
print(f"{'TOTAL':15s} {sum(len(v) for v in per_class.values()):>8d} ", end='')

# save
annotation = {'labels': CLASSES, 'database': database}
os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
with open(args.output, 'w') as f:
    json.dump(annotation, f, indent=2)

print(f"\nSplit summary:")
for k, v in split_counts.items():
    print(f"  {k:12s}: {v} videos")
print(f"\n✅ Annotation saved: {args.output}")
