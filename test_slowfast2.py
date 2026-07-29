import sys
import os
import json
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
    print("[WARN] scikit-learn not found. Confusion matrix and reports will be skipped.")

# ── ARGUMENT PARSING ──────────────────────────────────────────────────────────
def get_args():
    parser = argparse.ArgumentParser(description="Test SlowFast R50 (facebookresearch/pytorchvideo) model")
    parser.add_argument('--jpg_root', type=str, default='/kaggle/working/TEST_test_DATA_jpg_raw')
    parser.add_argument('--checkpoint', type=str, default='/kaggle/working/results_slowfast_ft/best_model.pth')
    parser.add_argument('--result_path', type=str, default='/kaggle/working/test_results_slowfast')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--n_workers', type=int, default=2)
    parser.add_argument('--fast_frames', type=int, default=32)
    parser.add_argument('--alpha', type=int, default=4,
                         help="Slow pathway = fast_frames[::alpha]. MUST match training (default 4).")
    parser.add_argument('--img_size', type=int, default=224)
    args, _ = parser.parse_known_args()
    return args

args = get_args()
jpg_root, checkpoint, result_path = args.jpg_root, args.checkpoint, args.result_path
batch_size, n_workers = args.batch_size, args.n_workers
fast_frames, alpha, img_size = args.fast_frames, args.alpha, args.img_size

os.makedirs(result_path, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Load checkpoint FIRST so we can pull the exact class order used in training ──
_ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
if isinstance(_ckpt, dict) and 'args' in _ckpt and 'classes' in _ckpt.get('args', {}):
    CLASSES = _ckpt['args']['classes'].split(',')
    print(f"[info] Loaded class order from checkpoint: {CLASSES}")
else:
    CLASSES = ['fight', 'unsafeClimb', 'unsafeJump', 'unsafeThrow', 'fall']
    print(f"[WARN] Checkpoint has no saved 'args.classes' -> falling back to hardcoded "
          f"default order {CLASSES}. Verify this matches how the model was actually trained!")
C2I   = {c: i for i, c in enumerate(CLASSES)}
N_CLS = len(CLASSES)

# ── Transforms & Dataset ──────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.45, 0.45, 0.45], [0.225, 0.225, 0.225]),  # Kinetics norm, matches training
])

class TestDataset(Dataset):
    """Mirrors SlowFastDataset's val-time sampling in train_slowfast_finetune.py:
    fast_frames are sampled first (deterministic, centered window), and the
    slow pathway is every alpha-th frame of THAT SAME window."""
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
            start = (total - n) // 2   # deterministic centered window (same as val in training)
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
                img = Image.new('RGB', (img_size, img_size), (0, 0, 0))
            frames.append(transform(img))
        fast_clip = torch.stack(frames, 0).permute(1, 0, 2, 3)  # [C, T_fast, H, W]

        slow_idx = torch.linspace(0, self.fast_frames - 1, self.fast_frames // self.alpha).long()
        slow_clip = torch.index_select(fast_clip, 1, slow_idx)  # [C, T_slow, H, W]

        return [slow_clip, fast_clip], label, c_name, v_name

def collate_fn(batch):
    return [torch.stack([b[0][0] for b in batch]), torch.stack([b[0][1] for b in batch])], \
           torch.tensor([b[1] for b in batch]), [b[2] for b in batch], [b[3] for b in batch]

# ── Model Loading ──────────────────────────────────────────────────────────────
def load_model(ckpt):
    state = ckpt.get('model_state', ckpt.get('state_dict', ckpt))
    model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=False)
    in_features = model.blocks[-1].proj.in_features
    model.blocks[-1].proj = nn.Linear(in_features, N_CLS)
    state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    print("✅ Loaded fine-tuned SlowFast-R50")
    return model.to(device).eval()

# ── Confusion matrix plot (same style as train_slowfast_finetune.py) ─────────
def save_confusion_matrix(y_true, y_pred, classes, out_path):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    cm_norm = cm.astype('float') / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, mat, title, fmt in zip(axes, [cm, cm_norm], ['Counts', 'Row-normalized'], ['d', '.2f']):
        im = ax.imshow(mat, cmap='Blues')
        ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=45, ha='right')
        ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
        ax.set_xlabel('Predicted'); ax.set_ylabel('Actual'); ax.set_title(title)
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, format(mat[i, j], fmt), ha='center', va='center',
                        color='white' if mat[i, j] > mat.max() / 2 else 'black')
        fig.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return cm

# ── Inference: pure argmax, no thresholding, no noclass ───────────────────────
def run_inference(model, loader):
    results = []
    with torch.no_grad():
        for inputs, labels, c_names, v_names in loader:
            inputs = [x.to(device) for x in inputs]
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            confs, preds = probs.max(dim=1)  # "dominant" class = argmax, that's all we need

            for j in range(len(labels)):
                pred, label = preds[j].item(), labels[j].item()
                results.append({
                    'video': v_names[j],
                    'actual': CLASSES[label],
                    'predicted': CLASSES[pred],
                    'conf': confs[j].item(),
                    'correct': pred == label,
                })
    return results

# ── MAIN ──────────────────────────────────────────────────────────────────────
try:
    ds = TestDataset(jpg_root, fast_frames, alpha)
    if len(ds) == 0:
        print("❌ No test samples found!")
    else:
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=n_workers, collate_fn=collate_fn)
        model = load_model(_ckpt)

        print(f"\n⏳ Running inference (argmax, no confidence gating)...")
        results = run_inference(model, loader)

        # 1. Video-wise table
        print(f"\n{'='*85}")
        print(f"{'VIDEO NAME':<30} | {'ACTUAL':<12} | {'PREDICTED':<12} | {'CONF':<6} | {'RESULT'}")
        print(f"{'-'*85}")
        for r in results:
            res_str = "✅" if r['correct'] else "❌"
            print(f"{r['video'][:30]:<30} | {r['actual']:<12} | {r['predicted']:<12} | {r['conf']:.2f} | {res_str}")
        print(f"{'='*85}\n")

        preds = np.array([C2I[r['predicted']] for r in results])
        labels = np.array([C2I[r['actual']] for r in results])
        correct = sum(r['correct'] for r in results)

        print(f"RESULTS SUMMARY:")
        print(f"  Total Videos    : {len(results)}")
        print(f"  Overall Accuracy: {correct}/{len(results)} ({correct/len(results)*100:.2f}%)")

        print(f"\nCLASS-WISE ACCURACY:")
        for i, cls in enumerate(CLASSES):
            cls_mask = (labels == i)
            if cls_mask.sum() > 0:
                cls_correct = ((preds == labels) & cls_mask).sum()
                print(f"  {cls:<15}: {cls_correct}/{cls_mask.sum()} ({cls_correct/cls_mask.sum()*100:.2f}%)")

        report_dict, cm = None, None
        if HAS_SK:
            print(f"\nCLASSIFICATION REPORT:")
            print(classification_report(labels, preds, target_names=CLASSES, zero_division=0))
            report_dict = classification_report(labels, preds, target_names=CLASSES,
                                                  zero_division=0, output_dict=True)
            cm = save_confusion_matrix(labels, preds, CLASSES, Path(result_path) / 'confusion_matrix.png')
            print(f"Saved confusion matrix -> {Path(result_path) / 'confusion_matrix.png'}")

        with open(Path(result_path) / 'test_results.json', 'w') as f:
            json.dump({
                'classes': CLASSES,
                'per_video_results': results,
                'accuracy': correct / len(results),
                'classification_report': report_dict,
                'confusion_matrix': cm.tolist() if cm is not None else None,
            }, f, indent=2)
        print(f"Saved detailed results -> {Path(result_path) / 'test_results.json'}")

except Exception:
    traceback.print_exc()
