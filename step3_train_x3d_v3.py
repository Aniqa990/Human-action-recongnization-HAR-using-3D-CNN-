"""
step3_train_x3d_v2.py  ─  Improved X3D Training (complete rewrite)
===================================================================

ANSWERS TO YOUR QUESTIONS — built into this script:

Q: Does the script do correct frame sampling?
   YES. Your clips are 2-6 s at 10 fps = 20-60 frames.
   Training : random temporal crop window of n_frames.
              If clip < n_frames, frames are looped/repeated to fill.
              Each epoch sees a DIFFERENT window of long clips.
   Validation: deterministic centre-crop, same window every time.

Q: Do I re-run prepare_annotation.py?
   NO. This script reads your dataset.json for labels and paths ONLY.
   It ignores the existing train/val split and builds its own grouped
   k-fold from scratch.

Q: Are weights computed correctly?
   YES, per fold.
   weight_i = N_train_total / (n_classes * count_i_in_THIS_fold)
   Then normalised so mean weight = 1.0.
   Classes with fewer training samples get higher weights.

Q: Does the script handle class imbalance?
   YES — three ways:
     1. Per-fold inverse-frequency class weights in CrossEntropyLoss.
     2. Label smoothing (reduces overconfidence on dominant classes).
     3. Random temporal crop means shorter/rarer classes get more
        diverse training signals per epoch.

Q: Should all layers be trained or just fine-tuned?
   TWO-PHASE fine-tuning:
   Phase 1 (first phase1_epochs): backbone FROZEN, only head trained.
     → Fast convergence, head adapts without corrupting Kinetics features.
   Phase 2 (remaining epochs): ALL layers unfrozen, low LR.
     → Backbone slowly adapts to your CCTV domain.

Q: AI video domain gap — is it a problem?
   YES. HD AI-generated video vs grainy CCTV footage is a real domain
   gap. This script applies mild quality-degradation augmentation:
     - Random Gaussian blur (simulates focus/motion blur)
     - Random small patch erasing (simulates JPEG compression blocks)
     - Strong colour jitter
   These make the model domain-invariant so it doesn't memorise
   "clean = AI video" vs "noisy = CCTV video".

USAGE:
    # 5-fold grouped CV (recommended)
    python step3_train_x3d_v2.py \
        --jpg_root    "H:/My Drive/FYP_DATA_jpg_raw" \
        --annotation  "H:/My Drive/FYP_DATA_jpg_raw/dataset.json" \
        --result_path "H:/My Drive/FYP_DATA_jpg_raw/results_x3d_v2" \
        --n_folds 5 --n_epochs 30 --batch_size 16

    # Kaggle T4
    python step3_train_x3d_v2.py \
        --jpg_root    /kaggle/working/FYP_DATA_jpg_raw \
        --annotation  /kaggle/working/FYP_DATA_jpg_raw/dataset.json \
        --result_path /kaggle/working/results_x3d_v2 \
        --n_folds 5 --n_epochs 25 --batch_size 8 --n_workers 2

    # Resume from fold 2 if folds 0 and 1 already finished
    python step3_train_x3d_v2.py ... --resume_from_fold 2
"""

import os, re, json, time, random, argparse
import numpy as np
from collections import Counter
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.transforms as T
from sklearn.model_selection import GroupKFold

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--jpg_root',         type=str,   required=True)
parser.add_argument('--annotation',       type=str,   required=True)
parser.add_argument('--result_path',      type=str,   required=True)
parser.add_argument('--n_folds',          type=int,   default=5)
parser.add_argument('--n_epochs',         type=int,   default=30)
parser.add_argument('--phase1_epochs',    type=int,   default=5,
                    help='Epochs with backbone frozen (head-only training)')
parser.add_argument('--batch_size',       type=int,   default=16)
parser.add_argument('--head_lr',          type=float, default=1e-3)
parser.add_argument('--full_lr',          type=float, default=1e-4)
parser.add_argument('--weight_decay',     type=float, default=1e-4)
parser.add_argument('--label_smoothing',  type=float, default=0.1)
parser.add_argument('--n_frames',         type=int,   default=16)
parser.add_argument('--img_size',         type=int,   default=160)
parser.add_argument('--x3d_variant',      type=str,   default='x3d_s',
                    choices=['x3d_xs', 'x3d_s', 'x3d_m'])
parser.add_argument('--n_workers',        type=int,   default=2)
parser.add_argument('--seed',             type=int,   default=42)
parser.add_argument('--resume_from_fold', type=int,   default=0,
                    help='Skip folds before this index. Default 0 = run all.')
args = parser.parse_args()

# ─────────────────────────────────────────────────────────────────────────────
# Constants & Setup
# ─────────────────────────────────────────────────────────────────────────────
CLASSES   = ['fight', 'unsafeClimb', 'unsafeJump', 'unsafeThrow', 'fall']
C2I       = {c: i for i, c in enumerate(CLASSES)}
N_CLASSES = len(CLASSES)

os.makedirs(args.result_path, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

print(f"\n{'='*65}")
print(f"  X3D TRAINING v2  —  Grouped {args.n_folds}-Fold CV")
print(f"{'='*65}")
print(f"  Device        : {device}")
print(f"  X3D variant   : {args.x3d_variant}")
print(f"  Total epochs  : {args.n_epochs}  "
      f"(head-only: {args.phase1_epochs}, "
      f"full: {args.n_epochs - args.phase1_epochs})")
print(f"  LR head/full  : {args.head_lr} / {args.full_lr}")
print(f"  Label smooth  : {args.label_smoothing}")
print(f"  Frames/clip   : {args.n_frames}")
print(f"  Image size    : {args.img_size}x{args.img_size}")
print(f"{'='*65}\n")

# ─────────────────────────────────────────────────────────────────────────────
# Source-ID extraction  ← prevents data leakage across augmented variants
# ─────────────────────────────────────────────────────────────────────────────
# Your augment_videos.py names files:
#   aug_HorizontalFlip_<original_stem>
#   aug_Brightness_<original_stem>
#   aug_GaussianNoise_<original_stem>
#   aug_TemporalJitter_<original_stem>
#   aug_Grayscale_<original_stem>
#   aug_RandomRotate_<original_stem>
# Originals have no prefix at all.
#
# We strip 'aug_<AnyWord>_' prefix to recover the ORIGINAL video stem.
# GroupKFold then uses this as the group so all augmented variants of
# the same source video always go into the SAME fold (train OR val,
# never split between them).

_AUG_RE = re.compile(r'^aug_[A-Za-z]+_(.+)$')

def extract_source_id(stem: str) -> str:
    m = _AUG_RE.match(stem)
    return m.group(1) if m else stem     # original video: returned unchanged


# ─────────────────────────────────────────────────────────────────────────────
# Frame sampling  ← handles 2-6 second clips correctly
# ─────────────────────────────────────────────────────────────────────────────

def sample_frames_train(files: list, n: int) -> list:
    """
    Random temporal crop for training.
    - Clip >= n frames : random contiguous window of n frames.
                         Different random window each epoch → model sees
                         more of the action over training.
    - Clip < n frames  : loop/repeat until we have n frames.
                         This is correct for 2s clips at 10fps (20 frames)
                         when n_frames=16 — the clip is long enough.
                         For very short clips (<16 frames) looping is the
                         standard practice (used by PyTorchVideo itself).
    """
    total = len(files)
    if total == 0:
        return []
    if total >= n:
        start = random.randint(0, total - n)
        return files[start: start + n]
    # loop
    repeated = []
    while len(repeated) < n:
        repeated.extend(files)
    return repeated[:n]


def sample_frames_val(files: list, n: int) -> list:
    """
    Deterministic centre-crop for validation — reproducible every run.
    """
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
# Transforms
# ─────────────────────────────────────────────────────────────────────────────

def make_train_transform(img_size: int) -> T.Compose:
    """
    Spatial augmentation per frame during training.
    GaussianBlur + RandomErasing simulate CCTV quality on HD AI videos,
    reducing the domain gap between your two video sources.
    """
    return T.Compose([
        T.Resize((img_size + 24, img_size + 24), antialias=True),
        T.RandomCrop(img_size),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
        T.RandomGrayscale(p=0.05),
        # Simulates focus blur / motion blur on HD footage
        T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.3),
        T.ToTensor(),
        # Simulates JPEG block artefacts (very small patches)
        T.RandomErasing(p=0.2, scale=(0.01, 0.05), ratio=(0.5, 2.0), value=0),
        T.Normalize([0.45, 0.45, 0.45], [0.225, 0.225, 0.225]),
    ])


def make_val_transform(img_size: int) -> T.Compose:
    return T.Compose([
        T.Resize((img_size, img_size), antialias=True),
        T.ToTensor(),
        T.Normalize([0.45, 0.45, 0.45], [0.225, 0.225, 0.225]),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Dataset — used in two ways:
#   1. Master scan (all videos, no transform) → extract groups/labels for kfold
#   2. FoldSubset (per-fold, correct transform) → actual training/val loading
# ─────────────────────────────────────────────────────────────────────────────

def scan_all_samples(jpg_root: str, annotation_path: str) -> list:
    """
    Returns list of (folder_path, label_idx, video_stem, source_id)
    for every video found in the annotation, regardless of subset.
    Does NOT load any frames — just collects metadata.
    """
    with open(annotation_path, 'r') as f:
        ann = json.load(f)

    samples = []
    for vid_stem, info in ann['database'].items():
        label_name = info['annotations']['label']
        label_idx  = C2I.get(label_name, -1)
        if label_idx < 0:
            continue
        for cand in [
            os.path.join(jpg_root, label_name, vid_stem),
            os.path.join(jpg_root, vid_stem),
        ]:
            if os.path.isdir(cand):
                src_id = extract_source_id(vid_stem)
                samples.append((cand, label_idx, vid_stem, src_id))
                break

    return samples


class FoldDataset(Dataset):
    """
    Dataset for one fold (train or val subset).
    Applies the correct transform (train augmentation vs val centre-crop).
    """
    def __init__(self, sample_list: list, is_train: bool):
        self.samples   = sample_list
        self.is_train  = is_train
        self.transform = (make_train_transform(args.img_size) if is_train
                          else make_val_transform(args.img_size))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        folder, label, _, _ = self.samples[idx]

        files = sorted(
            f for f in os.listdir(folder) if f.lower().endswith('.jpg')
        )
        selected = (sample_frames_train(files, args.n_frames) if self.is_train
                    else sample_frames_val(files, args.n_frames))

        if not selected:
            # Empty folder — black clip
            return torch.zeros(3, args.n_frames, args.img_size, args.img_size), label

        frames = []
        for fname in selected:
            try:
                img = Image.open(os.path.join(folder, fname)).convert('RGB')
            except Exception:
                img = Image.new('RGB', (args.img_size, args.img_size), (0, 0, 0))
            frames.append(self.transform(img))

        # [T, 3, H, W] → [3, T, H, W]
        clip = torch.stack(frames, dim=0).permute(1, 0, 2, 3)
        return clip, label


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

def build_x3d(variant: str):
    """
    Load X3D pretrained on Kinetics-400.  Replace head with n_classes output.
    Returns (model, arch_type) where arch_type is 'x3d' or 'r3d18'.
    """
    try:
        model = torch.hub.load(
            'facebookresearch/pytorchvideo', variant,
            pretrained=True, verbose=False,
        )
        in_feat = model.blocks[-1].proj.in_features
        model.blocks[-1].proj = nn.Linear(in_feat, N_CLASSES)
        print(f"  ✅ {variant.upper()} loaded — pretrained Kinetics-400 "
              f"({sum(p.numel() for p in model.parameters())/1e6:.1f}M params)")
        return model, 'x3d'
    except Exception as e:
        print(f"  [WARN] pytorchvideo hub failed: {e}")
        print(f"         Falling back to R3D-18")
        from torchvision.models.video import r3d_18, R3D_18_Weights
        model = r3d_18(weights=R3D_18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, N_CLASSES)
        print(f"  ✅ R3D-18 fallback loaded — pretrained Kinetics-400")
        return model, 'r3d18'


def freeze_backbone(model: nn.Module, arch: str):
    """Freeze all params; then unfreeze only the classification head."""
    for p in model.parameters():
        p.requires_grad = False
    if arch == 'x3d':
        for p in model.blocks[-1].proj.parameters():
            p.requires_grad = True
    else:
        for p in model.fc.parameters():
            p.requires_grad = True


def unfreeze_all(model: nn.Module):
    for p in model.parameters():
        p.requires_grad = True


# ─────────────────────────────────────────────────────────────────────────────
# Class weights  (per-fold, inverse-frequency)
# ─────────────────────────────────────────────────────────────────────────────

def compute_class_weights(train_labels: list) -> torch.Tensor:
    """
    Inverse-frequency weighting normalised to mean = 1.0.

    weight_i = N_total / (N_classes * count_i)

    Effect: if 'fight' has 400 train samples and 'unsafeThrow' has 200,
    unsafeThrow gets 2× the weight, so its misclassification costs more.
    This directly addresses class imbalance without oversampling.
    """
    counts  = Counter(train_labels)
    total   = sum(counts.values())
    w = torch.tensor(
        [total / (N_CLASSES * max(1, counts[i])) for i in range(N_CLASSES)],
        dtype=torch.float32,
    )
    return w / w.mean()   # normalise: mean weight = 1.0 (keeps loss scale stable)


# ─────────────────────────────────────────────────────────────────────────────
# Training epoch / validation epoch
# ─────────────────────────────────────────────────────────────────────────────

def run_epoch(model, loader, optimizer, criterion, is_train: bool):
    model.train() if is_train else model.eval()

    total_loss  = 0.0
    correct     = 0
    total       = 0
    cls_correct = [0] * N_CLASSES
    cls_total   = [0] * N_CLASSES

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for clips, labels in loader:
            clips  = clips.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if is_train:
                optimizer.zero_grad()

            out  = model(clips)
            loss = criterion(out, labels)

            if is_train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

            preds      = out.argmax(dim=1)
            correct   += (preds == labels).sum().item()
            total     += labels.size(0)
            total_loss += loss.item()

            for lbl, pred in zip(labels.cpu().tolist(), preds.cpu().tolist()):
                cls_total[lbl]   += 1
                cls_correct[lbl] += int(lbl == pred)

    acc     = correct / max(1, total)
    avg_los = total_loss / max(1, len(loader))
    per_cls = {CLASSES[i]: (cls_correct[i], cls_total[i])
               for i in range(N_CLASSES)}
    return avg_los, acc, per_cls


def print_per_class(per_cls: dict):
    for cls, (c, t) in per_cls.items():
        pct = 100.0 * c / max(1, t)
        mk  = '✅' if pct >= 70 else ('⚠️ ' if pct >= 50 else '❌')
        print(f"      {mk} {cls:15s}: {c:3d}/{t:3d}  ({pct:5.1f}%)")


# ─────────────────────────────────────────────────────────────────────────────
# One fold
# ─────────────────────────────────────────────────────────────────────────────

def train_fold(fold_idx: int, train_indices: list,
               val_indices: list, all_samples: list) -> float:

    fold_dir = os.path.join(args.result_path, f'fold_{fold_idx}')
    os.makedirs(fold_dir, exist_ok=True)

    # ── Leakage check ─────────────────────────────────────────────────────────
    train_srcs = {all_samples[i][3] for i in train_indices}
    val_srcs   = {all_samples[i][3] for i in val_indices}
    overlap    = train_srcs & val_srcs
    if overlap:
        print(f"\n  ⚠️  LEAKAGE: {len(overlap)} source IDs appear in BOTH "
              f"train and val! Examples: {list(overlap)[:3]}")
        print(f"      Check extract_source_id() against your file naming.")
    else:
        print(f"  ✅ No leakage — {len(train_srcs)} train sources, "
              f"{len(val_srcs)} val sources, 0 overlap")

    # ── Class balance ─────────────────────────────────────────────────────────
    train_labels = [all_samples[i][1] for i in train_indices]
    val_labels   = [all_samples[i][1] for i in val_indices]
    tr_counts    = Counter(train_labels)
    vl_counts    = Counter(val_labels)
    print(f"  Train: {len(train_indices)} | Val: {len(val_indices)}")
    print(f"  {'Class':15s}  {'Train':>6s}  {'Val':>5s}")
    for i, cls in enumerate(CLASSES):
        print(f"  {cls:15s}  {tr_counts.get(i,0):6d}  {vl_counts.get(i,0):5d}")

    # ── Dataloaders ───────────────────────────────────────────────────────────
    train_ds = FoldDataset([all_samples[i] for i in train_indices], is_train=True)
    val_ds   = FoldDataset([all_samples[i] for i in val_indices],   is_train=False)

    kw = dict(num_workers=args.n_workers, pin_memory=True,
              persistent_workers=(args.n_workers > 0))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  **kw)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, **kw)

    # ── Model ─────────────────────────────────────────────────────────────────
    print(f"\n  Building {args.x3d_variant}...")
    model, arch = build_x3d(args.x3d_variant)
    model = model.to(device)

    # ── Per-fold class weights ─────────────────────────────────────────────────
    weights = compute_class_weights(train_labels).to(device)
    print(f"  Per-fold class weights:")
    for i, cls in enumerate(CLASSES):
        print(f"      {cls:15s}: {weights[i].item():.3f}")

    criterion = nn.CrossEntropyLoss(weight=weights,
                                    label_smoothing=args.label_smoothing)

    best_val_acc = 0.0
    log_rows     = []

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 1: backbone FROZEN, head only
    # ──────────────────────────────────────────────────────────────────────────
    freeze_backbone(model, arch)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  ── Phase 1: head-only / backbone frozen ──")
    print(f"     Trainable params: {n_trainable:,}")

    opt1 = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.head_lr, weight_decay=args.weight_decay,
    )
    sch1 = CosineAnnealingLR(opt1, T_max=args.phase1_epochs,
                              eta_min=args.head_lr * 0.01)

    for ep in range(1, args.phase1_epochs + 1):
        t0 = time.time()
        tl, ta, _      = run_epoch(model, train_loader, opt1, criterion, True)
        vl, va, vl_cls = run_epoch(model, val_loader,   opt1, criterion, False)
        sch1.step()
        lr  = opt1.param_groups[0]['lr']
        print(f"  P1 [{ep:2d}/{args.phase1_epochs}] "
              f"Tr {tl:.4f}/{ta*100:.1f}%  Val {vl:.4f}/{va*100:.1f}%  "
              f"LR {lr:.2e}  {time.time()-t0:.0f}s")
        log_rows.append((ep, 'phase1', tl, ta, vl, va, lr))
        if va > best_val_acc:
            best_val_acc = va
            _save_ckpt(model, fold_dir, ep, fold_idx, va)

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 2: all layers, low LR
    # ──────────────────────────────────────────────────────────────────────────
    unfreeze_all(model)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    phase2_n    = args.n_epochs - args.phase1_epochs
    print(f"\n  ── Phase 2: full fine-tuning ({phase2_n} epochs) ──")
    print(f"     Trainable params: {n_trainable:,}")

    opt2 = torch.optim.AdamW(model.parameters(), lr=args.full_lr,
                              weight_decay=args.weight_decay)
    sch2 = CosineAnnealingLR(opt2, T_max=phase2_n,
                              eta_min=args.full_lr * 0.01)

    for ep in range(args.phase1_epochs + 1, args.n_epochs + 1):
        t0 = time.time()
        tl, ta, _      = run_epoch(model, train_loader, opt2, criterion, True)
        vl, va, vl_cls = run_epoch(model, val_loader,   opt2, criterion, False)
        sch2.step()
        lr = opt2.param_groups[0]['lr']
        print(f"  P2 [{ep:2d}/{args.n_epochs}] "
              f"Tr {tl:.4f}/{ta*100:.1f}%  Val {vl:.4f}/{va*100:.1f}%  "
              f"LR {lr:.2e}  {time.time()-t0:.0f}s")
        print_per_class(vl_cls)
        log_rows.append((ep, 'phase2', tl, ta, vl, va, lr))
        if va > best_val_acc:
            best_val_acc = va
            _save_ckpt(model, fold_dir, ep, fold_idx, va)
            print(f"  🏆 Best: {va*100:.2f}% → fold_{fold_idx}/best_model.pth")

    # ── Save log ──────────────────────────────────────────────────────────────
    with open(os.path.join(fold_dir, 'log.csv'), 'w') as f:
        f.write("epoch,phase,tr_loss,tr_acc,vl_loss,vl_acc,lr\n")
        for row in log_rows:
            f.write(",".join(
                str(round(x, 6) if isinstance(x, float) else x)
                for x in row) + "\n")

    print(f"\n  Fold {fold_idx} done — best val acc: {best_val_acc*100:.2f}%")
    return best_val_acc


def _save_ckpt(model, fold_dir, epoch, fold_idx, val_acc):
    torch.save({
        'epoch':      epoch,
        'fold':       fold_idx,
        'state_dict': model.state_dict(),
        'val_acc':    val_acc,
        'classes':    CLASSES,
        'args':       vars(args),
    }, os.path.join(fold_dir, 'best_model.pth'))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    print("Scanning all videos (ignoring existing train/val split)...")
    all_samples = scan_all_samples(args.jpg_root, args.annotation)

    if not all_samples:
        print("❌ No samples found. Check --jpg_root and --annotation.")
        exit(1)

    groups     = np.array([s[3] for s in all_samples])   # source_id per video
    labels_arr = np.array([s[1] for s in all_samples])
    indices    = np.arange(len(all_samples))

    # Summary
    unique_src = len(set(groups.tolist()))
    counts     = Counter(labels_arr.tolist())
    print(f"\n  Total video folders : {len(all_samples)}")
    print(f"  Unique source videos: {unique_src}")
    print(f"  Avg aug factor      : {len(all_samples)/unique_src:.1f}x per source")
    print(f"\n  Per-class totals:")
    for i, cls in enumerate(CLASSES):
        print(f"      {cls:15s}: {counts.get(i, 0):4d} folders")
    print()

    # Warn if any 'aug_' stem couldn't be stripped
    bad = [s[2] for s in all_samples
           if s[2].startswith('aug_') and extract_source_id(s[2]) == s[2]]
    if bad:
        print(f"  ⚠️  {len(bad)} aug_ videos not stripped correctly. "
              f"Examples: {bad[:3]}")
        print(f"     These will be treated as independent sources — fix naming.")

    # GroupKFold: all augmentations of one source → same fold guaranteed
    gkf    = GroupKFold(n_splits=args.n_folds)
    splits = list(gkf.split(indices, labels_arr, groups))

    fold_results = []
    for fold_i, (tr_idx, vl_idx) in enumerate(splits):

        if fold_i < args.resume_from_fold:
            print(f"[Skip fold {fold_i}]")
            continue

        print(f"\n{'─'*65}")
        print(f"  FOLD {fold_i + 1} / {args.n_folds}")
        print(f"{'─'*65}")

        acc = train_fold(fold_i, tr_idx.tolist(), vl_idx.tolist(), all_samples)
        fold_results.append(acc)

    # Summary
    print(f"\n{'='*65}")
    print(f"  K-FOLD SUMMARY")
    print(f"{'='*65}")
    for i, acc in enumerate(fold_results):
        print(f"  Fold {i}: {acc*100:.2f}%")
    mean_acc = float(np.mean(fold_results))
    std_acc  = float(np.std(fold_results))
    print(f"\n  Mean Val Acc : {mean_acc*100:.2f}%  ±  {std_acc*100:.2f}%")
    print(f"  Results dir  : {args.result_path}")
    print(f"{'='*65}\n")

    with open(os.path.join(args.result_path, 'cv_summary.json'), 'w') as f:
        json.dump({
            'fold_accs':    [float(a) for a in fold_results],
            'mean_val_acc': mean_acc,
            'std_val_acc':  std_acc,
            'args':         vars(args),
        }, f, indent=2)
    print("Saved cv_summary.json")
