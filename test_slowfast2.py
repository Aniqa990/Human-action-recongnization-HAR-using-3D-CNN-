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
    parser = argparse.ArgumentParser(description="Test SlowFast R50 (facebookresearch/pytorchvideo) model")
    parser.add_argument('--jpg_root', type=str, default='/kaggle/working/TEST_test_DATA_jpg_raw')
    parser.add_argument('--checkpoint', type=str, default='/kaggle/working/results_slowfast_ft/best_model.pth')
    parser.add_argument('--result_path', type=str, default='/kaggle/working/test_results_slowfast')
    parser.add_argument('--noclass_thresh', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--n_workers', type=int, default=2)
    # NOTE: slow_frames is no longer sampled independently -- it's derived from
    # fast_frames via --alpha, exactly like in train_slowfast_finetune.py.
    # Kept here only so old command lines that pass --slow_frames don't crash;
    # the value is ignored.
    parser.add_argument('--slow_frames', type=int, default=8, help="[ignored] kept for CLI compatibility")
    parser.add_argument('--fast_frames', type=int, default=32)
    parser.add_argument('--alpha', type=int, default=4,
                         help="Slow pathway = fast_frames[::alpha]. MUST match the value used "
                              "in training (default 4) or the pretrained/finetuned weights will "
                              "see misaligned input.")
    parser.add_argument('--img_size', type=int, default=224)
    args, _ = parser.parse_known_args()
    return args

args = get_args()
jpg_root, checkpoint, result_path = args.jpg_root, args.checkpoint, args.result_path
noclass_thresh, batch_size, n_workers = args.noclass_thresh, args.batch_size, args.n_workers
fast_frames, alpha, img_size = args.fast_frames, args.alpha, args.img_size

os.makedirs(result_path, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Load checkpoint FIRST so we can pull the exact class order used in training ──
# train_slowfast_finetune.py saves {'model_state': ..., 'args': vars(args), ...}
# and args['classes'] is the *exact* comma-separated list/order the model's
# output head was trained on. Hardcoding this list separately (as the old
# script did) risks a silent class-index mismatch with no error -- always
# derive it from the checkpoint instead.
_ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
_ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
if isinstance(_ckpt, dict) and 'args' in _ckpt and 'classes' in _ckpt.get('args', {}):
    CLASSES = _ckpt['args']['classes'].split(',')
    print(f"[info] Loaded class order from checkpoint: {CLASSES}")
else:
    # fallback only if testing a checkpoint that predates this args-saving format
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
    slow pathway is every alpha-th frame of THAT SAME window -- not sampled
    independently."""
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
    """ckpt is the already-loaded dict from torch.load (see above)."""
    state = ckpt.get('model_state', ckpt.get('state_dict', ckpt))
    model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=False)
    in_features = model.blocks[-1].proj.in_features
    model.blocks[-1].proj = nn.Linear(in_features, N_CLS)
    state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    print("✅ Loaded fine-tuned SlowFast-R50")
    return model.to(device).eval()

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
    ds = TestDataset(jpg_root, fast_frames, alpha)
    if len(ds) == 0:
        print("❌ No test samples found!")
    else:
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=n_workers, collate_fn=collate_fn)
        model = load_model(_ckpt)

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
        report_dict = None
        cm = None
        if HAS_SK:
            classified = (preds != -1)
            if classified.sum() > 0:
                print(f"\nREPORT (Excluding noclass):")
                print(classification_report(labels[classified], preds[classified], target_names=CLASSES, zero_division=0))
                report_dict = classification_report(labels[classified], preds[classified], target_names=CLASSES,
                                                      zero_division=0, output_dict=True)
                cm = confusion_matrix(labels[classified], preds[classified], labels=list(range(N_CLS)))

        # 5. Save results to result_path (the old script computed result_path but never wrote to it)
        with open(Path(result_path) / 'test_results.json', 'w') as f:
            json.dump({
                'classes': CLASSES,
                'per_video_results': results,
                'overall_accuracy': correct / len(results),
                'noclass_count': noclasses,
                'classification_report': report_dict,
                'confusion_matrix': cm.tolist() if cm is not None else None,
            }, f, indent=2)
        print(f"\nSaved detailed results -> {Path(result_path) / 'test_results.json'}")

except Exception:
    traceback.print_exc()
