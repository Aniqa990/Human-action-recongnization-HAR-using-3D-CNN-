"""
train_x3d_final.py
====================================================================
X3D fine-tuning for 5-class child-safety action recognition
(fight, unsafeClimb, unsafeJump, unsafeThrow, fall).

WHY THIS SCRIPT (vs. your step3_train_x3d_v5.py):

  * Model      : torch.hub 'facebookresearch/pytorchvideo' x3d_s,
                 pretrained=True (Kinetics-400). This is the correct,
                 equivalent alternative to the SlowFast repo's model-zoo
                 checkpoints for a plain classification fine-tune — you
                 do NOT need the full slowfast/detectron2 config stack
                 to get the same pretrained weights.

  * Data split : Read directly from dataset.json's 'training' /
                 'validation' subsets. step2_prepare_annotation.py
                 already grouped augmented variants by source video
                 before splitting, so there is no leakage risk left to
                 fix — re-doing GroupKFold on top (as v5 did) is
                 redundant.

  * Augmentation: LIGHT ONLY (random crop + horizontal flip). Your
                 clips are already augmented offline (Brightness,
                 GaussianNoise, RandomRotate, TemporalJitter,
                 Grayscale). Stacking heavy ColorJitter/GaussianBlur/
                 RandomErasing on top of that (as v5 did) compounds
                 distortions rather than adding new signal.

  * Class imbalance: Class-Balanced Loss (Cui et al., CVPR 2019,
                 "Class-Balanced Loss Based on Effective Number of
                 Samples") + Focal Loss (Lin et al., ICCV 2017).
                 Class-Balanced weighting is more correct than naive
                 inverse-frequency because it accounts for redundancy
                 among near-duplicate augmented clips. Focal loss
                 additionally down-weights EASY EXAMPLES automatically,
                 which is the principled fix for "don't let the model
                 coast on the class it already nails" (e.g. `fall`,
                 which pretrained backbones often already handle well)
                 — it responds to per-sample difficulty, not a label
                 you'd have to guess ahead of time. On top of that,
                 --fall_weight_multiplier lets you manually dial a
                 class's weight down further once you've SEEN the
                 confusion matrix from a first run.

  * Freeze/unfreeze: single training loop, not two separate optimizer
                 phases. Backbone parameters get requires_grad=False
                 for the first --phase1_epochs, then True after — the
                 SAME optimizer/scheduler run the whole time (with two
                 param groups: backbone @ full_lr, head @ head_lr).
                 This makes checkpoint/resume trivial: just restore
                 epoch, optimizer, scheduler, scaler state and keep
                 going. No "which phase was I in" bookkeeping.

  * Checkpointing: last_checkpoint.pth saved EVERY epoch (full resume
                 state incl. RNG). best_model.pth saved whenever val
                 macro-F1 improves (monitoring macro-F1, not accuracy,
                 is what you want with class imbalance).

  * Early stopping: patience on val macro-F1, reset when phase 2
                 begins (unfreezing changes the loss landscape, so an
                 old plateau shouldn't count against the new phase).

  * Metrics for your paper: per-epoch train/val loss+acc, best-epoch
                 per-class precision/recall/F1, confusion matrix,
                 macro/weighted F1, parameter count, and FLOPs via
                 fvcore (the same library the SlowFast repo itself
                 uses for FLOP counting, so your number is directly
                 comparable to published X3D FLOP figures).

USAGE (Kaggle, 2x T4):
    !pip install -q torch torchvision pytorchvideo fvcore scikit-learn

    !python train_x3d_final.py \
        --jpg_root    /kaggle/working/FYP_DATA_jpg_raw \
        --annotation  /kaggle/working/FYP_DATA_jpg_raw/dataset.json \
        --result_path /kaggle/working/results_x3d_final \
        --n_epochs 40 --phase1_epochs 5 --batch_size 16 --n_workers 4

RESUME after an interrupted run:
    !python train_x3d_final.py \
        --jpg_root ... --annotation ... --result_path /kaggle/working/results_x3d_final \
        --resume /kaggle/working/results_x3d_final/last_checkpoint.pth
====================================================================
"""

import os, json, time, random, argparse
import numpy as np
from collections import Counter
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
import torchvision.transforms as T
from sklearn.metrics import (precision_recall_fscore_support,
                              confusion_matrix, accuracy_score)

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--jpg_root',     type=str, required=True)
parser.add_argument('--annotation',   type=str, required=True)
parser.add_argument('--result_path',  type=str, required=True)

parser.add_argument('--classes', type=str, nargs='+',
                    default=['fight', 'unsafeClimb', 'unsafeJump', 'unsafeThrow', 'fall'],
                    help='Class list — must match what is in dataset.json labels. '
                         'Default matches your existing step2/step4 pipeline (no Normal).')

parser.add_argument('--n_epochs',        type=int,   default=40)
parser.add_argument('--phase1_epochs',   type=int,   default=5,
                    help='Epochs with backbone frozen (head-only) before unfreezing everything.')
parser.add_argument('--batch_size',      type=int,   default=16)
parser.add_argument('--head_lr',         type=float, default=1e-3)
parser.add_argument('--full_lr',         type=float, default=1e-4)
parser.add_argument('--weight_decay',    type=float, default=1e-4)
parser.add_argument('--label_smoothing', type=float, default=0.1)
parser.add_argument('--n_frames',        type=int,   default=13,
                    help='13 = official X3D-S clip length. Use 16 for x3d_m/xs consistency with your prior scripts.')
parser.add_argument('--img_size',        type=int,   default=160)
parser.add_argument('--x3d_variant',     type=str,   default='x3d_s',
                    choices=['x3d_xs', 'x3d_s', 'x3d_m'])
parser.add_argument('--n_workers',       type=int,   default=4)
parser.add_argument('--seed',            type=int,   default=42)

parser.add_argument('--focal_gamma', type=float, default=1.5,
                    help='Focal-loss focusing parameter. 0 = plain (class-balanced) cross-entropy.')
parser.add_argument('--cb_beta', type=float, default=0.999,
                    help='Class-Balanced Loss beta (Cui et al. 2019). Closer to 1 = stronger reweighting.')
parser.add_argument('--fall_weight_multiplier', type=float, default=1.0,
                    help="Extra multiplier applied to the 'fall' class weight AFTER class-balanced "
                         "weighting. Set e.g. 0.5 to deliberately reduce its share of the loss once "
                         "you've confirmed from a first run's confusion matrix that it is already "
                         "well-handled by the pretrained backbone.")

parser.add_argument('--patience', type=int, default=8,
                    help='Early-stopping patience in epochs, monitored on val macro-F1.')
parser.add_argument('--resume', type=str, default=None,
                    help='Path to last_checkpoint.pth to resume an interrupted run.')
parser.add_argument('--no_amp', action='store_true', help='Disable mixed precision.')
args = parser.parse_args()

CLASSES   = args.classes
C2I       = {c: i for i, c in enumerate(CLASSES)}
N_CLASSES = len(CLASSES)

os.makedirs(args.result_path, exist_ok=True)
device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_gpus      = torch.cuda.device_count()
use_amp     = (not args.no_amp) and torch.cuda.is_available()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

print(f"\n{'='*70}")
print(f"  X3D FINE-TUNING  —  {N_CLASSES}-class child-safety action recognition")
print(f"{'='*70}")
print(f"  Device        : {device}  |  GPUs available: {n_gpus}")
print(f"  X3D variant   : {args.x3d_variant}")
print(f"  Classes       : {CLASSES}")
print(f"  Epochs        : {args.n_epochs}  (backbone frozen for first {args.phase1_epochs})")
print(f"  LR head/full  : {args.head_lr} / {args.full_lr}")
print(f"  Frames/clip   : {args.n_frames}  |  Image size: {args.img_size}x{args.img_size}")
print(f"  Loss          : Class-Balanced (beta={args.cb_beta}) + Focal (gamma={args.focal_gamma})")
print(f"  Mixed prec.   : {use_amp}")
print(f"{'='*70}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Frame sampling
# ─────────────────────────────────────────────────────────────────────────────
def sample_frames_train(files, n):
    """Random contiguous window (loop if clip shorter than n)."""
    total = len(files)
    if total == 0:
        return []
    if total >= n:
        start = random.randint(0, total - n)
        return files[start:start + n]
    repeated = []
    while len(repeated) < n:
        repeated.extend(files)
    return repeated[:n]


def sample_frames_val(files, n):
    """Deterministic centre window — reproducible every run."""
    total = len(files)
    if total == 0:
        return []
    if total >= n:
        start = (total - n) // 2
        return files[start:start + n]
    repeated = []
    while len(repeated) < n:
        repeated.extend(files)
    return repeated[:n]


# ─────────────────────────────────────────────────────────────────────────────
# Transforms — LIGHT ONLY, since offline augmentation already exists
# ─────────────────────────────────────────────────────────────────────────────
def make_train_transform(img_size):
    return T.Compose([
        T.Resize((img_size + 16, img_size + 16), antialias=True),
        T.RandomCrop(img_size),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        T.Normalize([0.45, 0.45, 0.45], [0.225, 0.225, 0.225]),
    ])


def make_val_transform(img_size):
    return T.Compose([
        T.Resize((img_size, img_size), antialias=True),
        T.ToTensor(),
        T.Normalize([0.45, 0.45, 0.45], [0.225, 0.225, 0.225]),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Dataset — reads dataset.json subsets directly (already leak-free split)
# ─────────────────────────────────────────────────────────────────────────────
def load_samples(jpg_root, annotation_path, subset):
    with open(annotation_path, 'r') as f:
        ann = json.load(f)

    samples = []
    for vid_stem, info in ann['database'].items():
        if info['subset'] != subset:
            continue
        label_name = info['annotations']['label']
        label_idx  = C2I.get(label_name, -1)
        if label_idx < 0:
            continue
        for cand in [os.path.join(jpg_root, label_name, vid_stem),
                     os.path.join(jpg_root, vid_stem)]:
            if os.path.isdir(cand):
                samples.append((cand, label_idx))
                break
    return samples


class ClipDataset(Dataset):
    def __init__(self, samples, is_train):
        self.samples   = samples
        self.is_train  = is_train
        self.transform = make_train_transform(args.img_size) if is_train else make_val_transform(args.img_size)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        folder, label = self.samples[idx]
        files = sorted(f for f in os.listdir(folder) if f.lower().endswith('.jpg'))
        selected = (sample_frames_train(files, args.n_frames) if self.is_train
                    else sample_frames_val(files, args.n_frames))

        if not selected:
            return torch.zeros(3, args.n_frames, args.img_size, args.img_size), label

        frames = []
        for fname in selected:
            try:
                img = Image.open(os.path.join(folder, fname)).convert('RGB')
            except Exception:
                img = Image.new('RGB', (args.img_size, args.img_size), (0, 0, 0))
            frames.append(self.transform(img))

        clip = torch.stack(frames, dim=0).permute(1, 0, 2, 3)  # [3,T,H,W]
        return clip, label


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────
def build_x3d(variant):
    model = torch.hub.load('facebookresearch/pytorchvideo', variant,
                           pretrained=True, verbose=False)
    in_feat = model.blocks[-1].proj.in_features
    model.blocks[-1].proj = nn.Linear(in_feat, N_CLASSES)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Loaded {variant} (Kinetics-400 pretrained) — {n_params/1e6:.2f}M params")
    return model


def get_param_groups(model):
    """Separate head (classifier) params from backbone params."""
    head_params, backbone_params = [], []
    for name, p in model.named_parameters():
        if 'blocks.5.proj' in name or name.startswith('blocks.5.proj'):
            head_params.append(p)
        else:
            backbone_params.append(p)
    # Fallback in case naming differs across pytorchvideo versions:
    if len(head_params) == 0:
        head_params    = list(model.blocks[-1].proj.parameters())
        head_ids       = {id(p) for p in head_params}
        backbone_params = [p for p in model.parameters() if id(p) not in head_ids]
    return backbone_params, head_params


def set_backbone_trainable(model, trainable):
    backbone_params, _ = get_param_groups(model)
    for p in backbone_params:
        p.requires_grad = trainable


# ─────────────────────────────────────────────────────────────────────────────
# Class-Balanced + Focal Loss
#   Class-Balanced weighting: Cui et al., CVPR 2019
#   Focal loss:               Lin et al., ICCV 2017
# ─────────────────────────────────────────────────────────────────────────────
def compute_class_balanced_weights(train_labels, beta):
    counts = Counter(train_labels)
    weights = []
    for i in range(N_CLASSES):
        n_i = max(1, counts.get(i, 0))
        effective_num = 1.0 - beta ** n_i
        w = (1.0 - beta) / effective_num
        weights.append(w)
    w = torch.tensor(weights, dtype=torch.float32)
    w = w / w.mean()  # normalise: mean weight = 1.0, keeps loss scale stable

    fall_idx = C2I.get('fall', None)
    if fall_idx is not None and args.fall_weight_multiplier != 1.0:
        w[fall_idx] *= args.fall_weight_multiplier
        print(f"  Applied fall_weight_multiplier={args.fall_weight_multiplier} "
              f"→ 'fall' weight = {w[fall_idx].item():.3f}")

    return w


class FocalLoss(nn.Module):
    """
    Cross-entropy scaled by (1 - p_t)^gamma, so well-classified (easy)
    examples contribute less to the loss regardless of their class —
    this is what actually prevents the model from "coasting" on a class
    the pretrained backbone already finds easy (e.g. fall), because it
    responds to per-example confidence rather than a class label.
    """
    def __init__(self, weight=None, gamma=1.5, label_smoothing=0.0):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.weight,
                             label_smoothing=self.label_smoothing, reduction='none')
        pt = torch.exp(-ce.detach() if self.gamma == 0 else -ce)
        focal_term = (1 - pt) ** self.gamma if self.gamma > 0 else 1.0
        return (focal_term * ce).mean()


# ─────────────────────────────────────────────────────────────────────────────
# FLOPs / params (fvcore — same tool the SlowFast repo itself uses)
# ─────────────────────────────────────────────────────────────────────────────
def compute_flops_and_params(model):
    result = {'params_M': sum(p.numel() for p in model.parameters()) / 1e6}
    try:
        from fvcore.nn import FlopCountAnalysis, parameter_count
        dummy = torch.randn(1, 3, args.n_frames, args.img_size, args.img_size).to(device)
        model.eval()
        with torch.no_grad():
            flops = FlopCountAnalysis(model, dummy)
            flops.unsupported_ops_warnings(False)
            result['GFLOPs'] = flops.total() / 1e9
        model.train()
    except Exception as e:
        print(f"  [WARN] fvcore FLOP count failed ({e}). "
              f"Install with `pip install fvcore` for this figure.")
        result['GFLOPs'] = None
    return result


# ─────────────────────────────────────────────────────────────────────────────
# One epoch
# ─────────────────────────────────────────────────────────────────────────────
def run_epoch(model, loader, optimizer, criterion, scaler, is_train):
    model.train() if is_train else model.eval()

    total_loss, all_preds, all_labels = 0.0, [], []
    ctx = torch.enable_grad() if is_train else torch.no_grad()

    with ctx:
        for clips, labels in loader:
            clips  = clips.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with autocast():
                    out  = model(clips)
                    loss = criterion(out, labels)
            else:
                out  = model(clips)
                loss = criterion(out, labels)

            if is_train:
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()

            total_loss += loss.item()
            preds = out.argmax(dim=1).detach().cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / max(1, len(loader))
    acc      = accuracy_score(all_labels, all_preds) if all_labels else 0.0
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, labels=list(range(N_CLASSES)), zero_division=0)
    macro_f1    = f1.mean()
    weighted_f1 = np.average(f1, weights=support) if support.sum() > 0 else 0.0

    metrics = {
        'loss': avg_loss, 'acc': acc, 'macro_f1': macro_f1, 'weighted_f1': weighted_f1,
        'per_class': {CLASSES[i]: {'precision': float(precision[i]),
                                   'recall': float(recall[i]),
                                   'f1': float(f1[i]),
                                   'support': int(support[i])}
                      for i in range(N_CLASSES)},
        'all_preds': all_preds, 'all_labels': all_labels,
    }
    return metrics


def print_per_class(per_class):
    for cls, m in per_class.items():
        print(f"      {cls:15s}: P {m['precision']*100:5.1f}%  "
              f"R {m['recall']*100:5.1f}%  F1 {m['f1']*100:5.1f}%  (n={m['support']})")


# ─────────────────────────────────────────────────────────────────────────────
# Checkpointing
# ─────────────────────────────────────────────────────────────────────────────
def raw_state_dict(model):
    return model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()


def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, best_macro_f1,
                    epochs_no_improve, extra=None):
    ckpt = {
        'epoch': epoch,
        'state_dict': raw_state_dict(model),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'scaler': scaler.state_dict() if scaler is not None else None,
        'best_macro_f1': best_macro_f1,
        'epochs_no_improve': epochs_no_improve,
        'classes': CLASSES,
        'args': vars(args),
        'rng_state': {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, path)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("Loading dataset.json subsets...")
    train_samples = load_samples(args.jpg_root, args.annotation, 'training')
    val_samples   = load_samples(args.jpg_root, args.annotation, 'validation')

    if not train_samples:
        print("❌ No training samples found — check --jpg_root / --annotation."); return
    if not val_samples:
        print("❌ No validation samples found — check --jpg_root / --annotation."); return

    train_labels = [s[1] for s in train_samples]
    val_labels   = [s[1] for s in val_samples]
    print(f"  Train: {len(train_samples)}  |  Val: {len(val_samples)}")
    print(f"  {'Class':15s}  {'Train':>6s}  {'Val':>5s}")
    tr_counts, vl_counts = Counter(train_labels), Counter(val_labels)
    for i, cls in enumerate(CLASSES):
        print(f"  {cls:15s}  {tr_counts.get(i,0):6d}  {vl_counts.get(i,0):5d}")

    train_ds = ClipDataset(train_samples, is_train=True)
    val_ds   = ClipDataset(val_samples,   is_train=False)

    kw = dict(num_workers=args.n_workers, pin_memory=True,
              persistent_workers=(args.n_workers > 0))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  **kw)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, **kw)

    print(f"\nBuilding {args.x3d_variant}...")
    model = build_x3d(args.x3d_variant).to(device)

    # FLOPs / params BEFORE wrapping in DataParallel
    flop_info = compute_flops_and_params(model)
    print(f"  Params : {flop_info['params_M']:.2f} M")
    if flop_info['GFLOPs'] is not None:
        print(f"  GFLOPs : {flop_info['GFLOPs']:.2f}  (single {args.n_frames}-frame "
              f"{args.img_size}x{args.img_size} clip)")

    backbone_params, head_params = get_param_groups(model)
    weights = compute_class_balanced_weights(train_labels, args.cb_beta).to(device)
    print(f"  Class-balanced weights (beta={args.cb_beta}):")
    for i, cls in enumerate(CLASSES):
        print(f"      {cls:15s}: {weights[i].item():.3f}")

    criterion = FocalLoss(weight=weights, gamma=args.focal_gamma,
                          label_smoothing=args.label_smoothing)

    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': args.full_lr},
        {'params': head_params,     'lr': args.head_lr},
    ], weight_decay=args.weight_decay)

    scheduler = CosineAnnealingLR(optimizer, T_max=args.n_epochs, eta_min=1e-6)
    scaler = GradScaler(enabled=use_amp)

    if n_gpus > 1:
        print(f"  Using nn.DataParallel across {n_gpus} GPUs")
        model = nn.DataParallel(model)

    start_epoch       = 1
    best_macro_f1      = 0.0
    epochs_no_improve  = 0

    # ── Resume ────────────────────────────────────────────────────────────────
    if args.resume and os.path.exists(args.resume):
        print(f"\nResuming from: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        raw_state_dict(model).update(ckpt['state_dict'])  # no-op safety
        (model.module if isinstance(model, nn.DataParallel) else model).load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        if use_amp and ckpt.get('scaler'):
            scaler.load_state_dict(ckpt['scaler'])
        start_epoch       = ckpt['epoch'] + 1
        best_macro_f1      = ckpt.get('best_macro_f1', 0.0)
        epochs_no_improve  = ckpt.get('epochs_no_improve', 0)
        rng = ckpt.get('rng_state')
        if rng:
            random.setstate(rng['python'])
            np.random.set_state(rng['numpy'])
            torch.set_rng_state(rng['torch'])
            if rng['cuda'] is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rng['cuda'])
        print(f"  Resumed at epoch {start_epoch}  (best macro-F1 so far: {best_macro_f1*100:.2f}%)")

    log_rows = []
    log_path = os.path.join(args.result_path, 'log.csv')
    if not os.path.exists(log_path) or start_epoch == 1:
        with open(log_path, 'w') as f:
            f.write("epoch,phase,tr_loss,tr_acc,tr_macro_f1,vl_loss,vl_acc,vl_macro_f1,vl_weighted_f1,lr\n")

    print(f"\n{'─'*70}\n  TRAINING\n{'─'*70}")
    for epoch in range(start_epoch, args.n_epochs + 1):
        phase = 'phase1_frozen' if epoch <= args.phase1_epochs else 'phase2_full'

        # Entering phase 2 for the first time: unfreeze + reset early-stop patience
        if epoch == args.phase1_epochs + 1:
            print(f"\n  ── Unfreezing backbone (phase 2 begins) ──")
            set_backbone_trainable(model.module if isinstance(model, nn.DataParallel) else model, True)
            epochs_no_improve = 0
        elif epoch == start_epoch and phase == 'phase1_frozen':
            set_backbone_trainable(model.module if isinstance(model, nn.DataParallel) else model, False)
        elif epoch == start_epoch and phase == 'phase2_full':
            set_backbone_trainable(model.module if isinstance(model, nn.DataParallel) else model, True)

        t0 = time.time()
        tr = run_epoch(model, train_loader, optimizer, criterion, scaler, is_train=True)
        vl = run_epoch(model, val_loader,   optimizer, criterion, scaler, is_train=False)
        scheduler.step()
        lr_bb = optimizer.param_groups[0]['lr']

        print(f"  [{epoch:2d}/{args.n_epochs}] ({phase})  "
              f"Tr loss {tr['loss']:.4f} acc {tr['acc']*100:5.1f}% F1 {tr['macro_f1']*100:5.1f}%  |  "
              f"Val loss {vl['loss']:.4f} acc {vl['acc']*100:5.1f}% F1 {vl['macro_f1']*100:5.1f}%  "
              f"LR {lr_bb:.2e}  {time.time()-t0:.0f}s")
        print_per_class(vl['per_class'])

        log_rows.append((epoch, phase, tr['loss'], tr['acc'], tr['macro_f1'],
                         vl['loss'], vl['acc'], vl['macro_f1'], vl['weighted_f1'], lr_bb))
        with open(log_path, 'a') as f:
            f.write(",".join(str(round(x, 6) if isinstance(x, float) else x)
                             for x in log_rows[-1]) + "\n")

        improved = vl['macro_f1'] > best_macro_f1
        if improved:
            best_macro_f1 = vl['macro_f1']
            epochs_no_improve = 0
            cm = confusion_matrix(vl['all_labels'], vl['all_preds'], labels=list(range(N_CLASSES)))
            save_checkpoint(
                os.path.join(args.result_path, 'best_model.pth'),
                model, optimizer, scheduler, scaler, epoch, best_macro_f1, epochs_no_improve,
                extra={
                    'val_acc': vl['acc'], 'val_macro_f1': vl['macro_f1'],
                    'val_weighted_f1': vl['weighted_f1'], 'per_class': vl['per_class'],
                    'confusion_matrix': cm.tolist(), 'params_M': flop_info['params_M'],
                    'GFLOPs': flop_info['GFLOPs'],
                })
            print(f"  🏆 New best (val macro-F1 {best_macro_f1*100:.2f}%) → best_model.pth")
        else:
            epochs_no_improve += 1

        # Always save resumable "last" checkpoint
        save_checkpoint(os.path.join(args.result_path, 'last_checkpoint.pth'),
                        model, optimizer, scheduler, scaler, epoch, best_macro_f1, epochs_no_improve)

        if epochs_no_improve >= args.patience:
            print(f"\n  ⏹ Early stopping: no val macro-F1 improvement for "
                  f"{args.patience} epochs (best {best_macro_f1*100:.2f}%).")
            break

    # ── Final paper-ready summary from the BEST checkpoint ─────────────────────
    best_path = os.path.join(args.result_path, 'best_model.pth')
    if os.path.exists(best_path):
        best_ckpt = torch.load(best_path, map_location='cpu')
        summary = {
            'model': args.x3d_variant,
            'n_classes': N_CLASSES,
            'classes': CLASSES,
            'params_M': best_ckpt.get('params_M'),
            'GFLOPs': best_ckpt.get('GFLOPs'),
            'best_epoch': best_ckpt['epoch'],
            'val_accuracy': best_ckpt['val_acc'],
            'val_macro_f1': best_ckpt['val_macro_f1'],
            'val_weighted_f1': best_ckpt['val_weighted_f1'],
            'per_class': best_ckpt['per_class'],
            'confusion_matrix': best_ckpt['confusion_matrix'],
            'train_val_split': {'train_n': len(train_samples), 'val_n': len(val_samples)},
        }
        with open(os.path.join(args.result_path, 'final_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n{'='*70}\n  FINAL RESULT (best checkpoint, epoch {summary['best_epoch']})\n{'='*70}")
        print(f"  Params        : {summary['params_M']:.2f} M")
        if summary['GFLOPs'] is not None:
            print(f"  GFLOPs        : {summary['GFLOPs']:.2f}")
        print(f"  Val accuracy  : {summary['val_accuracy']*100:.2f}%")
        print(f"  Val macro-F1  : {summary['val_macro_f1']*100:.2f}%")
        print(f"  Val weighted-F1: {summary['val_weighted_f1']*100:.2f}%")
        print_per_class(summary['per_class'])
        print(f"\n  Full metrics saved to: {os.path.join(args.result_path, 'final_summary.json')}")
        print(f"  Per-epoch log saved to: {log_path}")
        print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
