"""
step4_test_x3d.py  ─  Updated for k-fold trained models
=========================================================

THREE TESTING MODES:

1. ENSEMBLE (recommended) — loads ALL fold models, averages softmax
   outputs, takes argmax. Best accuracy.

   python step4_test_x3d.py \
       --mode ensemble \
       --results_path /kaggle/working/results_x3d_v2 \
       --jpg_root /kaggle/working/FYP_TEST_JPG \
       --annotation /kaggle/working/FYP_TEST_JPG/dataset.json

2. BEST FOLD — automatically finds which fold had highest val acc
   and tests with that single model.

   python step4_test_x3d.py \
       --mode best_fold \
       --results_path /kaggle/working/results_x3d_v2 \
       --jpg_root /kaggle/working/FYP_TEST_JPG \
       --annotation /kaggle/working/FYP_TEST_JPG/dataset.json

3. SINGLE — test one specific checkpoint (original behaviour).

   python step4_test_x3d.py \
       --mode single \
       --model_path /kaggle/working/results_x3d_v2/fold_2/best_model.pth \
       --jpg_root /kaggle/working/FYP_TEST_JPG \
       --annotation /kaggle/working/FYP_TEST_JPG/dataset.json

IMPORTANT: CLASSES must match exactly what you trained with.
           This script uses 5 classes (no Normal) to match
           step3_train_x3d_v5.py.
"""

import os, json, argparse
import numpy as np
from PIL import Image
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--mode',         type=str, default='ensemble',
                    choices=['ensemble', 'best_fold', 'single'],
                    help='ensemble=average all folds (best), '
                         'best_fold=use highest val-acc fold, '
                         'single=one specific checkpoint')

# For ensemble / best_fold modes
parser.add_argument('--results_path', type=str, default=None,
                    help='Path to results_x3d_v2/ folder containing fold_0/, fold_1/ ...')
parser.add_argument('--n_folds',      type=int, default=5,
                    help='How many folds were trained (used to find fold dirs)')

# For single mode
parser.add_argument('--model_path',   type=str, default=None,
                    help='Path to a specific best_model.pth (single mode only)')

# Dataset
parser.add_argument('--jpg_root',     type=str, required=True)
parser.add_argument('--annotation',   type=str, default=None,
                    help='Path to dataset.json for test set. '
                         'If omitted, auto-discovers from jpg_root directory.')
parser.add_argument('--batch_size',   type=int, default=16)
parser.add_argument('--n_workers',    type=int, default=2)
parser.add_argument('--n_frames',     type=int, default=16)
parser.add_argument('--img_size',     type=int, default=160)
parser.add_argument('--x3d_variant',  type=str, default='x3d_s',
                    choices=['x3d_xs', 'x3d_s', 'x3d_m'])
parser.add_argument('--output_json',  type=str, default='test_results.json')
args = parser.parse_args()

# ─────────────────────────────────────────────────────────────────────────────
# CLASSES — must match training script exactly
# step3_train_x3d_v5.py uses 5 classes (Normal removed)
# ─────────────────────────────────────────────────────────────────────────────
CLASSES   = ['fight', 'unsafeClimb', 'unsafeJump', 'unsafeThrow', 'fall']
C2I       = {c: i for i, c in enumerate(CLASSES)}
N_CLASSES = len(CLASSES)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"\n{'='*70}")
print(f"  X3D TEST SCRIPT  —  mode: {args.mode}")
print(f"{'='*70}")
print(f"  Device     : {device}")
print(f"  Classes    : {CLASSES}")
print(f"  N classes  : {N_CLASSES}")
print(f"  Test root  : {args.jpg_root}")
print(f"  N frames   : {args.n_frames}")
print(f"  Image size : {args.img_size}x{args.img_size}")
print(f"{'='*70}\n")

# ─────────────────────────────────────────────────────────────────────────────
# Frame sampling  (centre-crop for test — deterministic)
# ─────────────────────────────────────────────────────────────────────────────
def sample_frames_test(files, n):
    total = len(files)
    if total == 0:
        return []
    if total >= n:
        start = (total - n) // 2
        return files[start: start + n]
    repeated = []
    while len(repeated) < n:
        repeated.extend(files)
    return repeated[:n]

# ─────────────────────────────────────────────────────────────────────────────
# Transform  (no augmentation at test time)
# ─────────────────────────────────────────────────────────────────────────────
test_transform = transforms.Compose([
    transforms.Resize((args.img_size, args.img_size), antialias=True),
    transforms.ToTensor(),
    transforms.Normalize([0.45, 0.45, 0.45], [0.225, 0.225, 0.225]),
])

# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
class TestDataset(Dataset):
    """
    Loads test videos.
    If annotation is provided: reads labels from JSON.
    If not: auto-discovers from jpg_root/ClassName/video_stem/ structure.
    Returns (clip, label_idx, video_id, label_name).
    """
    def __init__(self, jpg_root, annotation_path, n_frames, transform):
        self.n_frames  = n_frames
        self.transform = transform
        self.samples   = []   # (folder, label_idx, video_id, label_name)

        if annotation_path and os.path.exists(annotation_path):
            with open(annotation_path, 'r') as f:
                ann = json.load(f)
            for vid, info in ann['database'].items():
                lbl_name = info['annotations']['label']
                lbl_idx  = C2I.get(lbl_name, -1)
                if lbl_idx < 0:
                    # Class exists in JSON but not in our CLASSES list — skip
                    continue
                for cand in [
                    os.path.join(jpg_root, lbl_name, vid),
                    os.path.join(jpg_root, vid),
                ]:
                    if os.path.isdir(cand):
                        self.samples.append((cand, lbl_idx, vid, lbl_name))
                        break
        else:
            print("  [INFO] No annotation — auto-discovering from directory.")
            for cls in CLASSES:
                cls_dir = os.path.join(jpg_root, cls)
                if not os.path.isdir(cls_dir):
                    print(f"  [WARN] Not found: {cls_dir}")
                    continue
                for vid_id in sorted(os.listdir(cls_dir)):
                    vid_path = os.path.join(cls_dir, vid_id)
                    if not os.path.isdir(vid_path):
                        continue
                    jpgs = [f for f in os.listdir(vid_path)
                            if f.lower().endswith('.jpg')]
                    if not jpgs:
                        continue
                    self.samples.append((vid_path, C2I[cls], vid_id, cls))

        print(f"  Loaded {len(self.samples)} test videos")
        if len(self.samples) == 0:
            print("  [ERROR] No test videos found!")
            print(f"          jpg_root : {jpg_root}")
            print(f"          annotation: {annotation_path}")
            print(f"          Expected  : jpg_root/ClassName/video_id/*.jpg")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        folder, label, vid_id, lbl_name = self.samples[idx]
        files    = sorted(f for f in os.listdir(folder)
                          if f.lower().endswith('.jpg'))
        selected = sample_frames_test(files, self.n_frames)

        if not selected:
            return (torch.zeros(3, self.n_frames,
                                args.img_size, args.img_size),
                    label, vid_id, lbl_name)

        frames = []
        for fname in selected:
            try:
                img = Image.open(os.path.join(folder, fname)).convert('RGB')
            except Exception:
                img = Image.new('RGB', (args.img_size, args.img_size), (0,0,0))
            frames.append(self.transform(img))

        clip = torch.stack(frames, 0).permute(1, 0, 2, 3)  # [3,T,H,W]
        return clip, label, vid_id, lbl_name

# ─────────────────────────────────────────────────────────────────────────────
# Model builder  (matches training script exactly)
# ─────────────────────────────────────────────────────────────────────────────
def build_x3d(variant):
    try:
        model = torch.hub.load(
            'facebookresearch/pytorchvideo', variant,
            pretrained=False,   # weights loaded from checkpoint
            verbose=False,
        )
        in_feat = model.blocks[-1].proj.in_features
        model.blocks[-1].proj = nn.Linear(in_feat, N_CLASSES)
        return model, 'x3d'
    except Exception as e:
        print(f"  [WARN] pytorchvideo failed: {e} → using R3D-18")
        from torchvision.models.video import r3d_18
        model = r3d_18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, N_CLASSES)
        return model, 'r3d18'


def load_model_from_checkpoint(ckpt_path):
    """
    Build architecture, load weights from checkpoint.
    Verifies class count matches before loading.
    Returns model on device, ready for eval.
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)

    # Verify classes match
    ckpt_classes = ckpt.get('classes', None)
    if ckpt_classes and ckpt_classes != CLASSES:
        print(f"  ⚠️  Class mismatch!")
        print(f"      Checkpoint classes : {ckpt_classes}")
        print(f"      Script classes     : {CLASSES}")
        print(f"      Make sure CLASSES in this script matches training.")

    val_acc = ckpt.get('val_acc', 0.0)
    fold    = ckpt.get('fold',    '?')
    epoch   = ckpt.get('epoch',   '?')
    print(f"  Loading fold {fold}, epoch {epoch}, val_acc {val_acc*100:.2f}%")

    model, arch = build_x3d(args.x3d_variant)
    model.load_state_dict(ckpt['state_dict'])
    model = model.to(device)
    model.eval()
    return model, val_acc

# ─────────────────────────────────────────────────────────────────────────────
# Find checkpoint paths based on mode
# ─────────────────────────────────────────────────────────────────────────────
def get_checkpoint_paths():
    """
    Returns list of (ckpt_path, val_acc) tuples based on --mode.
    """
    if args.mode == 'single':
        if not args.model_path:
            raise ValueError("--model_path required for --mode single")
        ckpt = torch.load(args.model_path, map_location='cpu')
        return [(args.model_path, ckpt.get('val_acc', 0.0))]

    if not args.results_path:
        raise ValueError("--results_path required for --mode ensemble/best_fold")

    # Collect all fold checkpoints
    ckpt_paths = []
    for fold_i in range(args.n_folds):
        p = os.path.join(args.results_path, f'fold_{fold_i}', 'best_model.pth')
        if os.path.exists(p):
            ckpt = torch.load(p, map_location='cpu')
            val_acc = ckpt.get('val_acc', 0.0)
            ckpt_paths.append((p, val_acc))
            print(f"  Found fold_{fold_i}/best_model.pth  "
                  f"(val_acc {val_acc*100:.2f}%)")
        else:
            print(f"  [WARN] fold_{fold_i}/best_model.pth not found — skipping")

    if not ckpt_paths:
        raise FileNotFoundError(
            f"No best_model.pth files found in {args.results_path}/fold_*/")

    if args.mode == 'best_fold':
        # Keep only the checkpoint with the highest val acc
        best = max(ckpt_paths, key=lambda x: x[1])
        print(f"\n  Best fold: {best[0]}  (val_acc {best[1]*100:.2f}%)")
        return [best]

    # ensemble: return all
    print(f"\n  Ensemble: {len(ckpt_paths)} models")
    return ckpt_paths

# ─────────────────────────────────────────────────────────────────────────────
# Run inference — returns raw softmax probabilities per video
# ─────────────────────────────────────────────────────────────────────────────
def get_softmax_outputs(model, loader):
    """
    Returns dict: video_id → numpy array of shape [N_CLASSES] (softmax probs)
    Also returns ground-truth label per video_id.
    """
    model.eval()
    probs_dict  = {}   # video_id → [N_CLASSES] softmax
    labels_dict = {}   # video_id → int label
    names_dict  = {}   # video_id → label_name string

    with torch.no_grad():
        for clips, labels, video_ids, label_names in loader:
            clips  = clips.to(device, non_blocking=True)
            out    = model(clips)
            probs  = F.softmax(out, dim=1).cpu().numpy()

            for i, vid_id in enumerate(video_ids):
                probs_dict[vid_id]  = probs[i]
                labels_dict[vid_id] = int(labels[i])
                names_dict[vid_id]  = label_names[i]

    return probs_dict, labels_dict, names_dict

# ─────────────────────────────────────────────────────────────────────────────
# Evaluation metrics
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(avg_probs, labels_dict, names_dict):
    """
    avg_probs  : dict  video_id → [N_CLASSES] averaged softmax
    labels_dict: dict  video_id → int ground truth
    names_dict : dict  video_id → str label name
    Returns results dict.
    """
    all_preds       = []
    all_labels      = []
    all_video_ids   = []
    all_label_names = []
    cls_correct     = [0] * N_CLASSES
    cls_total       = [0] * N_CLASSES

    for vid_id, probs in avg_probs.items():
        pred  = int(np.argmax(probs))
        label = labels_dict[vid_id]
        all_preds.append(pred)
        all_labels.append(label)
        all_video_ids.append(vid_id)
        all_label_names.append(names_dict[vid_id])
        cls_total[label]   += 1
        cls_correct[label] += int(pred == label)

    overall_acc = sum(p == l for p, l in zip(all_preds, all_labels)) / max(1, len(all_preds))

    per_class = {}
    for i in range(N_CLASSES):
        pct = 100.0 * cls_correct[i] / max(1, cls_total[i])
        per_class[CLASSES[i]] = {
            'correct':  cls_correct[i],
            'total':    cls_total[i],
            'accuracy': pct,
        }

    return {
        'overall_accuracy': overall_acc,
        'per_class':        per_class,
        'all_preds':        all_preds,
        'all_labels':       all_labels,
        'all_video_ids':    all_video_ids,
        'all_label_names':  all_label_names,
    }


def print_confusion_matrix(preds, labels):
    cm = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
    for p, l in zip(preds, labels):
        cm[l, p] += 1
    header = "".join(f"{CLASSES[i][:10]:>12}" for i in range(N_CLASSES))
    print(f"\n{'Confusion Matrix':}")
    print(f"{'':>15}" + header)
    for i in range(N_CLASSES):
        row = "".join(f"{cm[i,j]:>12}" for j in range(N_CLASSES))
        print(f"  {CLASSES[i]:13s}" + row)
    return cm


def print_per_video(results):
    by_class = defaultdict(list)
    for pred, label, vid_id, lbl_name in zip(
            results['all_preds'], results['all_labels'],
            results['all_video_ids'], results['all_label_names']):
        by_class[lbl_name].append({
            'video_id':   vid_id,
            'pred_label': CLASSES[pred],
            'correct':    pred == label,
        })
    for cls in sorted(by_class):
        vids = by_class[cls]
        n_ok = sum(v['correct'] for v in vids)
        print(f"\n  {cls} ({n_ok}/{len(vids)}):")
        for v in vids:
            mk = '✅' if v['correct'] else '❌'
            print(f"    {mk} {v['video_id']:35s} → {v['pred_label']}")

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':

    # ── Find checkpoints ──────────────────────────────────────────────────────
    print("Finding checkpoints...")
    ckpt_paths = get_checkpoint_paths()

    # ── Build test dataloader ─────────────────────────────────────────────────
    print("\nLoading test dataset...")
    test_ds = TestDataset(
        jpg_root        = args.jpg_root,
        annotation_path = args.annotation,
        n_frames        = args.n_frames,
        transform       = test_transform,
    )
    if len(test_ds) == 0:
        exit(1)

    test_loader = DataLoader(
        test_ds,
        batch_size  = args.batch_size,
        shuffle     = False,
        num_workers = args.n_workers,
        pin_memory  = True,
        persistent_workers = (args.n_workers > 0),
    )

    # ── Run each model and accumulate softmax outputs ─────────────────────────
    print(f"\nRunning inference with {len(ckpt_paths)} model(s)...")
    accumulated_probs = {}   # video_id → sum of softmax arrays
    labels_dict = {}
    names_dict  = {}

    for ckpt_idx, (ckpt_path, _) in enumerate(ckpt_paths):
        print(f"\n  Model {ckpt_idx+1}/{len(ckpt_paths)}: {ckpt_path}")
        model, _ = load_model_from_checkpoint(ckpt_path)

        probs_dict, ldict, ndict = get_softmax_outputs(model, test_loader)

        # Accumulate (sum) softmax probabilities across models
        for vid_id, probs in probs_dict.items():
            if vid_id not in accumulated_probs:
                accumulated_probs[vid_id] = np.zeros(N_CLASSES)
            accumulated_probs[vid_id] += probs

        labels_dict.update(ldict)
        names_dict.update(ndict)

        # Free GPU memory between models
        del model
        torch.cuda.empty_cache()

    # Average the accumulated probabilities
    n_models = len(ckpt_paths)
    avg_probs = {vid_id: probs / n_models
                 for vid_id, probs in accumulated_probs.items()}

    # ── Evaluate ──────────────────────────────────────────────────────────────
    results = evaluate(avg_probs, labels_dict, names_dict)

    # ── Print results ─────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    mode_label = {'ensemble': 'ENSEMBLE', 'best_fold': 'BEST FOLD',
                  'single': 'SINGLE MODEL'}[args.mode]
    print(f"  TEST RESULTS  ({mode_label}, {n_models} model(s))")
    print(f"{'='*70}")
    print(f"\n  Overall Accuracy: {results['overall_accuracy']*100:.2f}%\n")

    print(f"  Per-Class Accuracy:")
    print(f"  {'-'*50}")
    for cls, m in results['per_class'].items():
        pct = m['accuracy']
        mk  = '✅' if pct >= 70 else ('⚠️ ' if pct >= 50 else '❌')
        print(f"  {mk} {cls:15s}: {m['correct']:3d}/{m['total']:3d} "
              f"({pct:6.2f}%)")

    cm = print_confusion_matrix(results['all_preds'], results['all_labels'])

    print(f"\n{'='*70}")
    print(f"  PER-VIDEO RESULTS")
    print(f"{'='*70}")
    print_per_video(results)

    # ── Save JSON ─────────────────────────────────────────────────────────────
    output_data = {
        'mode':             args.mode,
        'n_models':         n_models,
        'checkpoints':      [p for p, _ in ckpt_paths],
        'overall_accuracy': results['overall_accuracy'],
        'per_class':        results['per_class'],
        'confusion_matrix': cm.tolist(),
        'class_names':      CLASSES,
    }
    with open(args.output_json, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n  Results saved to: {args.output_json}")
    print(f"  Overall Acc: {results['overall_accuracy']*100:.2f}%")
    print(f"{'='*70}\n")
