"""
step3_train_x3d_v2.py  ─  Improved X3D Training with Resume Support
====================================================================

RESUME INSTRUCTIONS:
────────────────────
Scenario 1: Stopped mid-fold (most common)
    Check results_x3d_v2/fold_4/ for:
        last_epoch.pth   ← saved every epoch (overwrites previous)
        best_model.pth   ← saved only on val acc improvement

    Resume command:
        python step3_train_x3d_v2.py \
            --jpg_root ... --annotation ... --result_path ... \
            --resume_from_fold 4 \
            --resume_checkpoint "results_x3d_v2/fold_4/last_epoch.pth"

Scenario 2: Fold finished, stopped before next fold started
        python step3_train_x3d_v2.py \
            --jpg_root ... --annotation ... --result_path ... \
            --resume_from_fold 4

Scenario 3: No checkpoint exists for fold 4 (stopped before first save)
        python step3_train_x3d_v2.py \
            --jpg_root ... --annotation ... --result_path ... \
            --resume_from_fold 4
    (restarts fold 4 from scratch, folds 0-3 results are kept)

WHAT GETS SAVED EVERY EPOCH (last_epoch.pth):
    - model state_dict
    - optimizer state_dict
    - scheduler state_dict
    - current epoch number
    - current phase (phase1 or phase2)
    - best_val_acc so far in this fold
    - fold index
    - all args

This means if training stops at epoch 17 of fold 4, you resume from
epoch 18 with the exact same optimizer momentum/LR schedule state.
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
parser.add_argument('--jpg_root',           type=str, required=True)
parser.add_argument('--annotation',         type=str, default=None,
                    help='Path to dataset.json. If omitted, scans jpg_root '
                         'directory structure directly.')
parser.add_argument('--result_path',        type=str, required=True)
parser.add_argument('--n_folds',            type=int, default=5)
parser.add_argument('--n_epochs',           type=int, default=30)
parser.add_argument('--phase1_epochs',      type=int, default=5)
parser.add_argument('--batch_size',         type=int, default=16)
parser.add_argument('--head_lr',            type=float, default=1e-3)
parser.add_argument('--full_lr',            type=float, default=1e-4)
parser.add_argument('--weight_decay',       type=float, default=1e-4)
parser.add_argument('--label_smoothing',    type=float, default=0.1)
parser.add_argument('--n_frames',           type=int, default=16)
parser.add_argument('--img_size',           type=int, default=160)
parser.add_argument('--x3d_variant',        type=str, default='x3d_s',
                    choices=['x3d_xs', 'x3d_s', 'x3d_m'])
parser.add_argument('--n_workers',          type=int, default=2)
parser.add_argument('--seed',               type=int, default=42)
# ── Resume arguments ──────────────────────────────────────────────────────────
parser.add_argument('--resume_from_fold',   type=int, default=0,
                    help='Skip folds with index < this value. '
                         'Use 4 to start at fold 4.')
parser.add_argument('--resume_checkpoint',  type=str, default=None,
                    help='Path to last_epoch.pth to resume mid-fold. '
                         'Only used for the first fold that runs '
                         '(i.e. the fold = resume_from_fold).')
args = parser.parse_args()

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
CLASSES   = ['fight', 'Normal', 'unsafeClimb', 'unsafeJump', 'unsafeThrow', 'fall']
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
      f"(phase1={args.phase1_epochs}, "
      f"phase2={args.n_epochs - args.phase1_epochs})")
print(f"  LR head/full  : {args.head_lr} / {args.full_lr}")
print(f"  Label smooth  : {args.label_smoothing}")
print(f"  Frames/clip   : {args.n_frames}")
print(f"  Image size    : {args.img_size}x{args.img_size}")
if args.resume_from_fold > 0:
    print(f"  Resuming from : fold {args.resume_from_fold}")
if args.resume_checkpoint:
    print(f"  Checkpoint    : {args.resume_checkpoint}")
print(f"{'='*65}\n")

# ─────────────────────────────────────────────────────────────────────────────
# Source-ID extraction  (prevents data leakage between aug variants)
# ─────────────────────────────────────────────────────────────────────────────
_AUG_RE = re.compile(r'^aug_[A-Za-z]+_(.+)$')

def extract_source_id(stem: str) -> str:
    """
    'aug_Brightness_fight_001'  →  'fight_001'
    'fight_001'                 →  'fight_001'
    Ensures all augmented copies of the same video share one group key.
    """
    m = _AUG_RE.match(stem)
    return m.group(1) if m else stem

# ─────────────────────────────────────────────────────────────────────────────
# Frame sampling
# ─────────────────────────────────────────────────────────────────────────────

def sample_frames_train(files, n):
    total = len(files)
    if total == 0:
        return []
    if total >= n:
        start = random.randint(0, total - n)
        return files[start: start + n]
    repeated = []
    while len(repeated) < n:
        repeated.extend(files)
    return repeated[:n]


def sample_frames_val(files, n):
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

def make_train_transform(img_size):
    return T.Compose([
        T.Resize((img_size + 24, img_size + 24), antialias=True),
        T.RandomCrop(img_size),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
        T.RandomGrayscale(p=0.05),
        T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.3),
        T.ToTensor(),
        T.RandomErasing(p=0.2, scale=(0.01, 0.05), ratio=(0.5, 2.0), value=0),
        T.Normalize([0.45, 0.45, 0.45], [0.225, 0.225, 0.225]),
    ])


def make_val_transform(img_size):
    return T.Compose([
        T.Resize((img_size, img_size), antialias=True),
        T.ToTensor(),
        T.Normalize([0.45, 0.45, 0.45], [0.225, 0.225, 0.225]),
    ])

# ─────────────────────────────────────────────────────────────────────────────
# Sample discovery  (annotation file OR directory scan)
# ─────────────────────────────────────────────────────────────────────────────

def scan_all_samples(jpg_root, annotation_path=None):
    """
    Returns list of (folder_path, label_idx, video_stem, source_id).
    If annotation_path given: reads labels from JSON (subset field ignored).
    If None: discovers classes directly from jpg_root/ClassName/ structure.
    """
    samples = []

    if annotation_path and os.path.exists(annotation_path):
        with open(annotation_path, 'r') as f:
            ann = json.load(f)
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
                    samples.append((cand, label_idx, vid_stem,
                                    extract_source_id(vid_stem)))
                    break
    else:
        print("  [INFO] No annotation file — scanning directory structure.")
        for class_name in CLASSES:
            class_dir = os.path.join(jpg_root, class_name)
            if not os.path.isdir(class_dir):
                print(f"  [WARN] Not found: {class_dir}")
                continue
            for vid_stem in os.listdir(class_dir):
                folder = os.path.join(class_dir, vid_stem)
                if not os.path.isdir(folder):
                    continue
                jpgs = [f for f in os.listdir(folder) if f.endswith('.jpg')]
                if not jpgs:
                    continue
                samples.append((folder, C2I[class_name], vid_stem,
                                 extract_source_id(vid_stem)))

    return samples

# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class FoldDataset(Dataset):
    def __init__(self, sample_list, is_train):
        self.samples   = sample_list
        self.is_train  = is_train
        self.transform = (make_train_transform(args.img_size) if is_train
                          else make_val_transform(args.img_size))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        folder, label, _, _ = self.samples[idx]
        files    = sorted(f for f in os.listdir(folder)
                          if f.lower().endswith('.jpg'))
        selected = (sample_frames_train(files, args.n_frames) if self.is_train
                    else sample_frames_val(files, args.n_frames))
        if not selected:
            return torch.zeros(3, args.n_frames,
                               args.img_size, args.img_size), label
        frames = []
        for fname in selected:
            try:
                img = Image.open(os.path.join(folder, fname)).convert('RGB')
            except Exception:
                img = Image.new('RGB', (args.img_size, args.img_size), (0,0,0))
            frames.append(self.transform(img))
        return torch.stack(frames, 0).permute(1, 0, 2, 3), label

# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

def build_x3d(variant):
    try:
        model = torch.hub.load('facebookresearch/pytorchvideo', variant,
                               pretrained=True, verbose=False)
        in_feat = model.blocks[-1].proj.in_features
        model.blocks[-1].proj = nn.Linear(in_feat, N_CLASSES)
        total_p = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"  ✅ {variant.upper()} loaded — Kinetics-400 ({total_p:.1f}M params)")
        return model, 'x3d'
    except Exception as e:
        print(f"  [WARN] pytorchvideo failed: {e}  → using R3D-18 fallback")
        from torchvision.models.video import r3d_18, R3D_18_Weights
        model = r3d_18(weights=R3D_18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, N_CLASSES)
        return model, 'r3d18'


def freeze_backbone(model, arch):
    for p in model.parameters():
        p.requires_grad = False
    head_params = (model.blocks[-1].proj.parameters() if arch == 'x3d'
                   else model.fc.parameters())
    for p in head_params:
        p.requires_grad = True


def unfreeze_all(model):
    for p in model.parameters():
        p.requires_grad = True

# ─────────────────────────────────────────────────────────────────────────────
# Class weights
# ─────────────────────────────────────────────────────────────────────────────

def compute_class_weights(train_labels):
    counts = Counter(train_labels)
    total  = sum(counts.values())
    w = torch.tensor(
        [total / (N_CLASSES * max(1, counts[i])) for i in range(N_CLASSES)],
        dtype=torch.float32)
    return w / w.mean()

# ─────────────────────────────────────────────────────────────────────────────
# Epoch runner
# ─────────────────────────────────────────────────────────────────────────────

def run_epoch(model, loader, optimizer, criterion, is_train):
    model.train() if is_train else model.eval()
    total_loss = 0.0
    correct = 0
    total   = 0
    cls_c   = [0] * N_CLASSES
    cls_t   = [0] * N_CLASSES

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
            preds      = out.argmax(1)
            correct   += (preds == labels).sum().item()
            total     += labels.size(0)
            total_loss += loss.item()
            for l, p in zip(labels.cpu().tolist(), preds.cpu().tolist()):
                cls_t[l] += 1
                cls_c[l] += int(l == p)

    acc     = correct / max(1, total)
    avg_los = total_loss / max(1, len(loader))
    per_cls = {CLASSES[i]: (cls_c[i], cls_t[i]) for i in range(N_CLASSES)}
    return avg_los, acc, per_cls


def print_per_class(per_cls):
    for cls, (c, t) in per_cls.items():
        pct = 100.0 * c / max(1, t)
        mk  = '✅' if pct >= 70 else ('⚠️ ' if pct >= 50 else '❌')
        print(f"      {mk} {cls:15s}: {c:3d}/{t:3d}  ({pct:5.1f}%)")

# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_last_epoch(path, model, opt, sch, epoch, phase,
                    fold_idx, best_val_acc, log_rows):
    """
    Saves everything needed to resume mid-fold from the NEXT epoch.
    Overwrites the same file every epoch so disk usage stays small.
    """
    torch.save({
        'epoch':        epoch,
        'phase':        phase,            # 'phase1' or 'phase2'
        'fold':         fold_idx,
        'state_dict':   model.state_dict(),
        'optimizer':    opt.state_dict(),
        'scheduler':    sch.state_dict(),
        'best_val_acc': best_val_acc,
        'log_rows':     log_rows,
        'classes':      CLASSES,
        'args':         vars(args),
    }, path)


def save_best_model(path, model, epoch, fold_idx, val_acc):
    torch.save({
        'epoch':      epoch,
        'fold':       fold_idx,
        'state_dict': model.state_dict(),
        'val_acc':    val_acc,
        'classes':    CLASSES,
        'args':       vars(args),
    }, path)

# ─────────────────────────────────────────────────────────────────────────────
# Single fold training  (with mid-fold resume)
# ─────────────────────────────────────────────────────────────────────────────

def train_fold(fold_idx, train_indices, val_indices, all_samples,
               resume_ckpt_path=None):
    """
    resume_ckpt_path: path to last_epoch.pth if resuming mid-fold.
                      Pass None to start fold from scratch.
    """
    fold_dir      = os.path.join(args.result_path, f'fold_{fold_idx}')
    last_ep_path  = os.path.join(fold_dir, 'last_epoch.pth')
    best_mod_path = os.path.join(fold_dir, 'best_model.pth')
    log_csv_path  = os.path.join(fold_dir, 'log.csv')
    os.makedirs(fold_dir, exist_ok=True)

    # ── Leakage check ─────────────────────────────────────────────────────────
    train_srcs = {all_samples[i][3] for i in train_indices}
    val_srcs   = {all_samples[i][3] for i in val_indices}
    overlap    = train_srcs & val_srcs
    if overlap:
        print(f"\n  ⚠️  LEAKAGE: {len(overlap)} source IDs in both train & val!")
        print(f"      Examples: {list(overlap)[:3]}")
    else:
        print(f"  ✅ No leakage — {len(train_srcs)} train sources, "
              f"{len(val_srcs)} val sources")

    # ── Class balance ─────────────────────────────────────────────────────────
    train_labels = [all_samples[i][1] for i in train_indices]
    val_labels   = [all_samples[i][1] for i in val_indices]
    tr_c = Counter(train_labels)
    vl_c = Counter(val_labels)
    print(f"  Train: {len(train_indices)} | Val: {len(val_indices)}")
    print(f"  {'Class':15s}  {'Train':>6}  {'Val':>5}")
    for i, cls in enumerate(CLASSES):
        print(f"  {cls:15s}  {tr_c.get(i,0):6d}  {vl_c.get(i,0):5d}")

    # ── Dataloaders ───────────────────────────────────────────────────────────
    kw = dict(num_workers=args.n_workers, pin_memory=True,
              persistent_workers=(args.n_workers > 0))
    train_loader = DataLoader(
        FoldDataset([all_samples[i] for i in train_indices], True),
        batch_size=args.batch_size, shuffle=True, **kw)
    val_loader = DataLoader(
        FoldDataset([all_samples[i] for i in val_indices], False),
        batch_size=args.batch_size, shuffle=False, **kw)

    # ── Model ─────────────────────────────────────────────────────────────────
    print(f"\n  Building {args.x3d_variant}...")
    model, arch = build_x3d(args.x3d_variant)
    model = model.to(device)

    # ── Loss ──────────────────────────────────────────────────────────────────
    weights   = compute_class_weights(train_labels).to(device)
    print(f"  Class weights:")
    for i, cls in enumerate(CLASSES):
        print(f"      {cls:15s}: {weights[i].item():.3f}")
    criterion = nn.CrossEntropyLoss(weight=weights,
                                    label_smoothing=args.label_smoothing)

    # ── Decide where to start ─────────────────────────────────────────────────
    # If a resume checkpoint is provided, load it and figure out which
    # epoch/phase we left off at.  Otherwise start from epoch 1 phase1.
    resume_epoch    = 0       # last completed epoch (0 = none)
    resume_phase    = None    # 'phase1' or 'phase2'
    best_val_acc    = 0.0
    log_rows        = []
    loaded_opt_state = None
    loaded_sch_state = None

    ckpt_to_load = resume_ckpt_path or (
        last_ep_path if os.path.exists(last_ep_path) else None
    )

    if ckpt_to_load and os.path.exists(ckpt_to_load):
        print(f"\n  Loading checkpoint: {ckpt_to_load}")
        ckpt = torch.load(ckpt_to_load, map_location=device)
        model.load_state_dict(ckpt['state_dict'])
        resume_epoch     = ckpt['epoch']           # last COMPLETED epoch
        resume_phase     = ckpt['phase']
        best_val_acc     = ckpt.get('best_val_acc', 0.0)
        log_rows         = ckpt.get('log_rows', [])
        loaded_opt_state = ckpt['optimizer']
        loaded_sch_state = ckpt['scheduler']
        print(f"  Resumed — last completed epoch: {resume_epoch} "
              f"({resume_phase}), best val acc so far: "
              f"{best_val_acc*100:.2f}%")
    else:
        print(f"\n  Starting fold {fold_idx} from scratch.")

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 1 — head only, backbone frozen
    # Runs epochs 1 .. phase1_epochs
    # Skip entirely if we already finished phase1 (resume_phase == 'phase2'
    # or resume_epoch >= phase1_epochs)
    # ─────────────────────────────────────────────────────────────────────────
    phase1_done = (resume_epoch >= args.phase1_epochs)

    if not phase1_done:
        freeze_backbone(model, arch)
        n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n  ── Phase 1: head-only ({args.phase1_epochs} epochs, "
              f"{n_tr:,} trainable params) ──")

        opt1 = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.head_lr, weight_decay=args.weight_decay)
        sch1 = CosineAnnealingLR(opt1, T_max=args.phase1_epochs,
                                 eta_min=args.head_lr * 0.01)

        # Restore optimizer/scheduler state if we're mid-phase1
        if loaded_opt_state and resume_phase == 'phase1':
            opt1.load_state_dict(loaded_opt_state)
            sch1.load_state_dict(loaded_sch_state)
            print(f"  Restored phase1 optimizer & scheduler state.")

        start_ep = resume_epoch + 1
        for ep in range(start_ep, args.phase1_epochs + 1):
            t0 = time.time()
            tl, ta, _      = run_epoch(model, train_loader, opt1, criterion, True)
            vl, va, vl_cls = run_epoch(model, val_loader,   opt1, criterion, False)
            sch1.step()
            lr = opt1.param_groups[0]['lr']
            print(f"  P1 [{ep:2d}/{args.phase1_epochs}] "
                  f"Tr {tl:.4f}/{ta*100:.1f}%  "
                  f"Val {vl:.4f}/{va*100:.1f}%  "
                  f"LR {lr:.2e}  {time.time()-t0:.0f}s")
            log_rows.append((ep, 'phase1', tl, ta, vl, va, lr))

            if va > best_val_acc:
                best_val_acc = va
                save_best_model(best_mod_path, model, ep, fold_idx, va)

            # Save last-epoch checkpoint after every epoch
            save_last_epoch(last_ep_path, model, opt1, sch1,
                            ep, 'phase1', fold_idx, best_val_acc, log_rows)

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 2 — all layers, low LR
    # Runs epochs phase1_epochs+1 .. n_epochs
    # ─────────────────────────────────────────────────────────────────────────
    unfreeze_all(model)
    phase2_n = args.n_epochs - args.phase1_epochs
    n_tr     = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  ── Phase 2: full fine-tuning ({phase2_n} epochs, "
          f"{n_tr:,} trainable params) ──")

    opt2 = torch.optim.AdamW(model.parameters(), lr=args.full_lr,
                             weight_decay=args.weight_decay)
    sch2 = CosineAnnealingLR(opt2, T_max=phase2_n,
                             eta_min=args.full_lr * 0.01)

    # Restore optimizer/scheduler if we're resuming mid-phase2
    if loaded_opt_state and resume_phase == 'phase2':
        opt2.load_state_dict(loaded_opt_state)
        sch2.load_state_dict(loaded_sch_state)
        print(f"  Restored phase2 optimizer & scheduler state.")

    # Figure out the first epoch of phase2 we still need to run
    if resume_phase == 'phase2':
        start_ep = resume_epoch + 1
    else:
        start_ep = args.phase1_epochs + 1

    for ep in range(start_ep, args.n_epochs + 1):
        t0 = time.time()
        tl, ta, _      = run_epoch(model, train_loader, opt2, criterion, True)
        vl, va, vl_cls = run_epoch(model, val_loader,   opt2, criterion, False)
        sch2.step()
        lr = opt2.param_groups[0]['lr']
        print(f"  P2 [{ep:2d}/{args.n_epochs}] "
              f"Tr {tl:.4f}/{ta*100:.1f}%  "
              f"Val {vl:.4f}/{va*100:.1f}%  "
              f"LR {lr:.2e}  {time.time()-t0:.0f}s")
        print_per_class(vl_cls)
        log_rows.append((ep, 'phase2', tl, ta, vl, va, lr))

        if va > best_val_acc:
            best_val_acc = va
            save_best_model(best_mod_path, model, ep, fold_idx, va)
            print(f"  🏆 Best: {va*100:.2f}% → fold_{fold_idx}/best_model.pth")

        # Save last-epoch checkpoint after every epoch
        save_last_epoch(last_ep_path, model, opt2, sch2,
                        ep, 'phase2', fold_idx, best_val_acc, log_rows)

    # ── Save training log ─────────────────────────────────────────────────────
    with open(log_csv_path, 'w') as f:
        f.write("epoch,phase,tr_loss,tr_acc,vl_loss,vl_acc,lr\n")
        for row in log_rows:
            f.write(",".join(
                str(round(x, 6) if isinstance(x, float) else x)
                for x in row) + "\n")

    print(f"\n  Fold {fold_idx} done — best val acc: {best_val_acc*100:.2f}%")
    return best_val_acc

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    print("Scanning dataset...")
    all_samples = scan_all_samples(args.jpg_root, args.annotation)

    if not all_samples:
        print("No samples found. Check --jpg_root and --annotation.")
        exit(1)

    groups    = np.array([s[3] for s in all_samples])
    labels    = np.array([s[1] for s in all_samples])
    indices   = np.arange(len(all_samples))
    uniq_src  = len(set(groups.tolist()))

    print(f"\n  Total video folders : {len(all_samples)}")
    print(f"  Unique source videos: {uniq_src}")
    print(f"  Avg aug factor      : {len(all_samples)/uniq_src:.1f}x")
    counts = Counter(labels.tolist())
    for i, cls in enumerate(CLASSES):
        print(f"  {cls:15s}: {counts.get(i,0):4d}")

    gkf    = GroupKFold(n_splits=args.n_folds)
    splits = list(gkf.split(indices, labels, groups))

    fold_results = []

    for fold_i, (tr_idx, vl_idx) in enumerate(splits):

        if fold_i < args.resume_from_fold:
            # Try to recover the best acc from already-saved checkpoint
            best_path = os.path.join(args.result_path,
                                     f'fold_{fold_i}', 'best_model.pth')
            if os.path.exists(best_path):
                ckpt = torch.load(best_path, map_location='cpu')
                saved_acc = ckpt.get('val_acc', 0.0)
                fold_results.append(saved_acc)
                print(f"[Skip fold {fold_i} — loaded saved acc "
                      f"{saved_acc*100:.2f}%]")
            else:
                print(f"[Skip fold {fold_i} — no saved checkpoint found]")
            continue

        print(f"\n{'─'*65}")
        print(f"  FOLD {fold_i + 1} / {args.n_folds}")
        print(f"{'─'*65}")

        # Pass the resume checkpoint only for the FIRST fold we actually run
        ckpt_path = (args.resume_checkpoint
                     if fold_i == args.resume_from_fold
                     else None)

        acc = train_fold(fold_i, tr_idx.tolist(), vl_idx.tolist(),
                         all_samples, resume_ckpt_path=ckpt_path)
        fold_results.append(acc)

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
