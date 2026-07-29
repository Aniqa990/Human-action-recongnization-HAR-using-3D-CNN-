import sys
import os
import json
import time
import traceback
import argparse
import numpy as np
from PIL import Image
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Fix NameError: HAS_SK ─────────────────────────────────────────────────────
try:
    from sklearn.metrics import confusion_matrix, classification_report
    HAS_SK = True
except ImportError:
    HAS_SK = False
    print("[WARN] scikit-learn not found. Confusion matrix and reports will be skipped.")

# ── ARGUMENT PARSING ──────────────────────────────────────────────────────────
def get_args():
    parser = argparse.ArgumentParser(description="Test SlowFast / Two-Stream model")
    parser.add_argument('--jpg_root', type=str, default='/kaggle/working/TEST_DATA_jpg_raw')
    parser.add_argument('--checkpoint', type=str, default='/kaggle/working/results_slowfast/best_model.pth')
    parser.add_argument('--result_path', type=str, default='/kaggle/working/test_results_slowfast')
    parser.add_argument('--noclass_thresh', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--n_workers', type=int, default=2)
    parser.add_argument('--slow_frames', type=int, default=8)
    parser.add_argument('--fast_frames', type=int, default=32)
    parser.add_argument('--img_size', type=int, default=224)
    args, _ = parser.parse_known_args()
    return args

args = get_args()
jpg_root, checkpoint, result_path = args.jpg_root, args.checkpoint, args.result_path
noclass_thresh, batch_size, n_workers = args.noclass_thresh, args.batch_size, args.n_workers
slow_frames, fast_frames, img_size = args.slow_frames, args.fast_frames, args.img_size

CLASSES   = ['fight', 'fall', 'unsafeThrow', 'unsafeClimb', 'unsafeJump']
C2I       = {c: i for i, c in enumerate(CLASSES)}
N_CLS     = len(CLASSES)

os.makedirs(result_path, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Transforms & Dataset ──────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.45, 0.45, 0.45], [0.225, 0.225, 0.225]),
])

class TestDataset(Dataset):
    def __init__(self, jpg_root, slow_frames=8, fast_frames=32):
        self.slow_frames, self.fast_frames, self.samples = slow_frames, fast_frames, []
        root = Path(jpg_root)
        for cls in CLASSES:
            cls_dir = root / cls
            if not cls_dir.exists(): continue
            for vid_dir in sorted(cls_dir.iterdir()):
                if not vid_dir.is_dir(): continue
                self.samples.append((str(vid_dir), C2I[cls], cls, vid_dir.name))
        print(f"Total test samples: {len(self.samples)}\n")

    def _load_frames(self, vid_dir, n):
        files = sorted(list(Path(vid_dir).glob("*.jpg")))
        if not files: return torch.zeros(3, n, img_size, img_size)
        indices = np.linspace(0, len(files)-1, n, dtype=int)
        frames = [transform(Image.open(files[i]).convert('RGB')) for i in indices]
        return torch.stack(frames, 0).permute(1, 0, 2, 3)

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        v_dir, label, c_name, v_name = self.samples[idx]
        return [self._load_frames(v_dir, slow_frames), self._load_frames(v_dir, fast_frames)], label, c_name, v_name

def collate_fn(batch):
    return [torch.stack([b[0][0] for b in batch]), torch.stack([b[0][1] for b in batch])], \
           torch.tensor([b[1] for b in batch]), [b[2] for b in batch], [b[3] for b in batch]

# ── Model Loading (Same Logic) ────────────────────────────────────────────────
class TwoStreamModel(nn.Module):
    def __init__(self, n_classes=5):
        super().__init__()
        from torchvision.models.video import r3d_18
        slow, fast = r3d_18(weights=None), r3d_18(weights=None)
        self.slow_encoder, self.fast_encoder = nn.Sequential(*list(slow.children())[:-1]), nn.Sequential(*list(fast.children())[:-1])
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.5), nn.Linear(1024, 512), nn.ReLU(inplace=True), nn.Dropout(0.3), nn.Linear(512, n_classes))
    def forward(self, inputs):
        sf = self.slow_encoder(inputs[0]).flatten(1)
        ff = self.fast_encoder(inputs[1]).flatten(1)
        return self.classifier(torch.cat([sf, ff], dim=1))

def load_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get('state_dict', ckpt)
    try:
        m = torch.hub.load('facebookresearch/pytorchvideo:main', 'slowfast_r50', pretrained=False)
        m.blocks[-1].proj = nn.Linear(m.blocks[-1].proj.in_features, N_CLS)
        m.load_state_dict({k.replace('module.',''): v for k,v in state.items()}, strict=False)
        print("✅ Loaded SlowFast-R50")
    except:
        m = TwoStreamModel(n_classes=N_CLS)
        m.load_state_dict({k.replace('module.',''): v for k,v in state.items()}, strict=True)
        print("✅ Loaded Two-Stream R3D-18")
    return m.to(device).eval()

# ── Inference ─────────────────────────────────────────────────────────────────
def run_inference(model, loader, threshold):
    results = []
    with torch.no_grad():
        for i, (inputs, labels, c_names, v_names) in enumerate(loader):
            inputs = [x.to(device) for x in inputs]
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            confs, preds = probs.max(dim=1)

            for j in range(len(labels)):
                conf, pred, label = confs[j].item(), preds[j].item(), labels[j].item()
                final_pred = pred if conf >= threshold else -1
                results.append({
                    'video': v_names[j],
                    'actual': CLASSES[label],
                    'predicted': CLASSES[final_pred] if final_pred != -1 else 'noclass',
                    'conf': conf,
                    'correct': final_pred == label
                })
    return results

# ── MAIN ──────────────────────────────────────────────────────────────────────
try:
    ds = TestDataset(jpg_root, slow_frames, fast_frames)
    if len(ds) == 0:
        print("❌ No test samples found!")
    else:
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=n_workers, collate_fn=collate_fn)
        model = load_model(checkpoint)
        
        print(f"\n⏳ Running inference (threshold: {noclass_thresh})...")
        results = run_inference(model, loader, noclass_thresh)

        # 1. Video-wise Output Table
        print(f"\n{'='*85}")
        print(f"{'VIDEO NAME':<30} | {'ACTUAL':<12} | {'PREDICTED':<12} | {'CONF':<6} | {'RESULT'}")
        print(f"{'-'*85}")
        for r in results:
            res_str = "✅" if r['correct'] else "❌"
            print(f"{r['video'][:30]:<30} | {r['actual']:<12} | {r['predicted']:<12} | {r['conf']:.2f} | {res_str}")
        print(f"{'='*85}\n")

        # 2. Summary Metrics
        preds = np.array([C2I.get(r['predicted'], -1) for r in results])
        labels = np.array([C2I[r['actual']] for r in results])
        
        correct = sum(r['correct'] for r in results)
        noclasses = sum(1 for r in results if r['predicted'] == 'noclass')
        
        print(f"RESULTS SUMMARY:")
        print(f"  Total Videos      : {len(results)}")
        print(f"  Overall Correct   : {correct}/{len(results)} ({correct/len(results)*100:.2f}%)")
        print(f"  noclass triggered : {noclasses}")

        # 3. Class-wise Accuracy
        print(f"\nCLASS-WISE ACCURACY:")
        for i, cls in enumerate(CLASSES):
            cls_mask = (labels == i)
            if cls_mask.sum() > 0:
                cls_correct = ((preds == labels) & cls_mask).sum()
                print(f"  {cls:<15}: {cls_correct}/{cls_mask.sum()} ({cls_correct/cls_mask.sum()*100:.2f}%)")

        # 4. Scikit-Learn Reports (If available)
        if HAS_SK:
            classified = (preds != -1)
            if classified.sum() > 0:
                print(f"\nREPORT (Excluding noclass):")
                print(classification_report(labels[classified], preds[classified], target_names=CLASSES, zero_division=0))

except Exception:
    traceback.print_exc()
