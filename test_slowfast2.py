import sys
import os
import json
import time
import traceback
import argparse
import numpy as np
from PIL import Image
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from sklearn.metrics import confusion_matrix, classification_report
    HAS_SK = True
except ImportError:
    HAS_SK = False
    print("[WARN] scikit-learn not found.")

def get_args():
    parser = argparse.ArgumentParser(description="Test SlowFast R50 model")
    parser.add_argument('--jpg_root', type=str, default='/kaggle/working/TEST_test_DATA_jpg_raw')
    parser.add_argument('--checkpoint', type=str, default='/kaggle/working/results_slowfast_ft/best_model.pth')
    parser.add_argument('--result_path', type=str, default='/kaggle/working/test_results_slowfast')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--n_workers', type=int, default=0) # 0 worker prevents Kaggle hangs
    parser.add_argument('--fast_frames', type=int, default=32)
    parser.add_argument('--alpha', type=int, default=4)
    parser.add_argument('--img_size', type=int, default=224)
    args, _ = parser.parse_known_args()
    return args

args = get_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs(args.result_path, exist_ok=True)

# ── Load Checkpoint & Derive Class List ─────────────────────────────────────
_ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
if isinstance(_ckpt, dict) and 'args' in _ckpt and 'classes' in _ckpt.get('args', {}):
    CLASSES = _ckpt['args']['classes'].split(',')
    print(f"[info] Loaded class order from checkpoint: {CLASSES}")
else:
    CLASSES = ['fight', 'unsafeClimb', 'unsafeThrow', 'fall']

C2I = {c: i for i, c in enumerate(CLASSES)}
N_CLS = len(CLASSES)

transform = transforms.Compose([
    transforms.Resize((args.img_size, args.img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.45, 0.45, 0.45], [0.225, 0.225, 0.225]),
])

class TestDataset(Dataset):
    def __init__(self, jpg_root, fast_frames=32, alpha=4):
        self.fast_frames, self.alpha, self.samples = fast_frames, alpha, []
        root = Path(jpg_root)
        for cls in CLASSES:
            cls_dir = root / cls
            if not cls_dir.exists():
                continue
            for vid_dir in sorted(cls_dir.iterdir()):
                if not vid_dir.is_dir():
                    continue
                self.samples.append((str(vid_dir), C2I[cls], cls, vid_dir.name))
        print(f"Total test samples: {len(self.samples)}\n")

    def _sample_indices(self, total):
        n = self.fast_frames
        if total <= 0:
            return np.zeros(n, dtype=int)
        if total >= n:
            start = (total - n) // 2
            return np.arange(start, start + n)
        return np.linspace(0, total - 1, n).round().astype(int)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        v_dir, label, c_name, v_name = self.samples[idx]
        files = sorted(Path(v_dir).glob("*.jpg"))
        total = len(files)
        fast_idx = self._sample_indices(total)

        frames = []
        for i in fast_idx:
            i = int(np.clip(i, 0, max(total - 1, 0)))
            try:
                img = Image.open(files[i]).convert('RGB')
            except Exception:
                img = Image.new('RGB', (args.img_size, args.img_size), (0, 0, 0))
            frames.append(transform(img))
        fast_clip = torch.stack(frames, 0).permute(1, 0, 2, 3)

        slow_idx = torch.linspace(0, self.fast_frames - 1, self.fast_frames // self.alpha).long()
        slow_clip = torch.index_select(fast_clip, 1, slow_idx)

        return [slow_clip, fast_clip], label, c_name, v_name

def collate_fn(batch):
    return [torch.stack([b[0][0] for b in batch]), torch.stack([b[0][1] for b in batch])], \
           torch.tensor([b[1] for b in batch]), [b[2] for b in batch], [b[3] for b in batch]

def load_model(ckpt):
    state = ckpt.get('model_state', ckpt.get('state_dict', ckpt))
    model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=False)
    in_features = model.blocks[-1].proj.in_features
    model.blocks[-1].proj = nn.Linear(in_features, N_CLS)
    state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()

def run_inference(model, loader):
    results = []
    with torch.no_grad():
        for i, (inputs, labels, c_names, v_names) in enumerate(loader):
            inputs = [x.to(device) for x in inputs]
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            confs, preds = probs.max(dim=1) # Pure argmax

            for j in range(len(labels)):
                conf, pred, label = confs[j].item(), preds[j].item(), labels[j].item()
                results.append({
                    'video': v_names[j],
                    'actual': CLASSES[label],
                    'predicted': CLASSES[pred],
                    'conf': conf,
                    'correct': pred == label
                })
    return results

try:
    ds = TestDataset(args.jpg_root, args.fast_frames, args.alpha)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers, collate_fn=collate_fn)
    model = load_model(_ckpt)

    print("\n⏳ Running standard Top-1 inference...")
    results = run_inference(model, loader)

    print(f"\n{'='*85}")
    print(f"{'VIDEO NAME':<30} | {'ACTUAL':<12} | {'PREDICTED':<12} | {'CONF':<6} | {'RESULT'}")
    print(f"{'-'*85}")
    for r in results:
        res_str = "✅" if r['correct'] else "❌"
        print(f"{r['video'][:30]:<30} | {r['actual']:<12} | {r['predicted']:<12} | {r['conf']:.2f} | {res_str}")
    print(f"{'='*85}\n")

    y_true = np.array([C2I[r['actual']] for r in results])
    y_pred = np.array([C2I[r['predicted']] for r in results])
    correct = sum(r['correct'] for r in results)

    print(f"OVERALL RESULTS:")
    print(f"  Total Test Videos : {len(results)}")
    print(f"  Top-1 Accuracy    : {correct}/{len(results)} ({correct/len(results)*100:.2f}%)\n")

    if HAS_SK:
        print("CLASSIFICATION REPORT:")
        print(classification_report(y_true, y_pred, target_names=CLASSES, zero_division=0))

except Exception:
    traceback.print_exc()
