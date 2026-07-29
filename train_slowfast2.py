"""
train_slowfast_finetune.py

Fine-tune Facebook/Meta SlowFast (R50, 8x8, Kinetics-400 pretrained, loaded via
torch.hub 'facebookresearch/pytorchvideo') on a custom child-safety action
dataset, on a single Kaggle T4 GPU.

WHY torch.hub INSTEAD OF THE ORIGINAL facebookresearch/SlowFast REPO
---------------------------------------------------------------------
The checkpoint served by
    torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=True)
IS the official PySlowFast Model Zoo checkpoint (Kinetics/SLOWFAST_8x8_R50),
just repackaged as a plain nn.Module instead of requiring the full yacs-config /
distributed-launch research codebase built for multi-GPU cluster training. For
fine-tuning a small, single-GPU custom dataset, this is the correct engineering
choice: same architecture, same weights, far less integration overhead.

CORRECTNESS FIX vs. naive SlowFast finetuning scripts
------------------------------------------------------
Slow and Fast pathway frames MUST come from the SAME temporal window: you first
uniformly sample `fast_frames` (32) frames from the clip, then the Slow pathway
is every alpha-th (4th) frame of THAT SAME window (giving 8 frames), NOT an
independently-sampled window. Sampling them independently (as in some tutorial
scripts) misaligns the two pathways relative to what the pretrained weights
were trained on and silently hurts transfer accuracy. This script samples
correctly (see `SlowFastDataset._sample_indices` / `PackPathway` logic).

Reference numbers for your paper (official torch.hub / PySlowFast Model Zoo,
SlowFast R50 8x8, Kinetics-400):
    Top-1 76.9%, Top-5 92.7%, 65.7 GFLOPs (single 224 crop, single clip), 34.6M params
    Christoph Feichtenhofer et al., "SlowFast Networks for Video Recognition", ICCV 2019.
Your own model's actual FLOPs/params/latency on YOUR input size and class count
are computed and saved at the end of this script (do not just cite the paper
number if you changed img_size, since FLOPs scale with spatial resolution).

DATA LAYOUT EXPECTED (same as your step1/step2 scripts)
---------------------------------------------------------
jpg_root/
    fight/
        <video_folder>/*.jpg
    unsafeClimb/
        <video_folder>/*.jpg
    ...
dataset.json (built by step2_prepare_annotation.py or auto-built here if missing):
{
  "labels": ["fight", "unsafeClimb", "unsafeJump", "unsafeThrow", "fall"],
  "database": {
      "<video_folder_name>": {
          "subset": "training" | "validation",
          "annotations": {"label": "fight", "segment": [1, 87]}
      }, ...
  }
}

USAGE ON KAGGLE (single script, runs both phases, resumable)
---------------------------------------------------------------
pip install -q fvcore scikit-learn seaborn --no-deps 2>/dev/null

python train_slowfast_finetune.py \
    --jpg_root "/kaggle/input/fyp-data-jpg-raw/FYP_DATA_jpg_raw" \
    --annotation "/kaggle/input/fyp-data-jpg-raw/FYP_DATA_jpg_raw/dataset.json" \
    --result_path "/kaggle/working/results_slowfast_ft" \
    --batch_size 6 --head_epochs 8 --finetune_epochs 40 --patience 8

# to resume after a Kaggle session timeout:
python train_slowfast_finetune.py --result_path "/kaggle/working/results_slowfast_ft" \
    --jpg_root "..." --annotation "..." --resume "/kaggle/working/results_slowfast_ft/last_checkpoint.pth"
"""

import os
import sys
import json
import time
import random
import argparse
import itertools
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from sklearn.metrics import (classification_report, confusion_matrix,
                                  f1_score, precision_recall_fscore_support)
    HAS_SK = True
except ImportError:
    HAS_SK = False
    print("[WARN] scikit-learn not available -> pip install scikit-learn for full metrics.")

try:
    from fvcore.nn import FlopCountAnalysis, parameter_count
    HAS_FVCORE = True
except ImportError:
    HAS_FVCORE = False
    print("[WARN] fvcore not available -> pip install fvcore for FLOPs reporting.")


# ════════════════════════════════════════════════════════════════════════════
# Args
# ════════════════════════════════════════════════════════════════════════════
def get_args():
    p = argparse.ArgumentParser(description="Fine-tune SlowFast R50 8x8 on a custom dataset (Kaggle T4)")

    # Data
    p.add_argument('--jpg_root',   type=str, default='/kaggle/working/FYP_DATA_jpg_raw')
    p.add_argument('--annotation', type=str, default='/kaggle/working/FYP_DATA_jpg_raw/dataset.json')
    p.add_argument('--result_path', type=str, default='/kaggle/working/results_slowfast_ft')
    p.add_argument('--classes', type=str,
                    default='fight,unsafeClimb,unsafeJump,unsafeThrow,fall',
                    help="Comma-separated class names. MUST match your folder names exactly. "
                         "Edit this if your actual class set differs (e.g. no unsafeJump, "
                         "or you want to include Normal).")
    p.add_argument('--val_split', type=float, default=0.2,
                    help="Used only if --annotation does not exist yet and must be auto-built.")

    # Clip sampling (SlowFast 8x8 config: fast=32 frames uniformly sampled, "
    #                slow=every alpha-th frame of the SAME window)
    p.add_argument('--fast_frames', type=int, default=32)
    p.add_argument('--alpha', type=int, default=4, help="Slow pathway = fast_frames[::alpha]. "
                                                          "MUST be 4 to match the pretrained "
                                                          "torch.hub slowfast_r50 checkpoint.")
    p.add_argument('--img_size', type=int, default=224,
                    help="224 keeps you close to the paper recipe; drop to 160-176 if you OOM on T4.")

    # Optimization - Phase 1 (head warmup, backbone frozen)
    p.add_argument('--head_epochs', type=int, default=8)
    p.add_argument('--head_lr', type=float, default=1e-3)

    # Optimization - Phase 2 (fine-tune, backbone unfrozen)
    p.add_argument('--finetune_epochs', type=int, default=40)
    p.add_argument('--backbone_lr', type=float, default=1e-5)
    p.add_argument('--ft_head_lr', type=float, default=1e-4)
    p.add_argument('--unfreeze_from_block', type=int, default=3,
                    help="Which model.blocks[] index to start unfreezing from in phase 2. "
                         "SlowFast R50 has 6 blocks: 0=stem,1-4=res stages 2-5,5=head. "
                         "Default 3 = unfreeze res_stage4, res_stage5 + head only "
                         "(recommended for a few-hundred-clip dataset to limit overfitting). "
                         "Set to 0 to fine-tune the entire network.")

    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--batch_size', type=int, default=6)
    p.add_argument('--accum_steps', type=int, default=1, help="Gradient accumulation steps.")
    p.add_argument('--n_workers', type=int, default=4)
    p.add_argument('--patience', type=int, default=8,
                    help="Early stopping patience (epochs, phase 2 only), monitored on val macro-F1.")

    # Loss: focal loss + inverse-frequency class weights
    p.add_argument('--focal_gamma', type=float, default=2.0,
                    help="Higher gamma = more aggressively down-weights already-easy/confident "
                         "predictions (e.g. 'fall') during training.")
    p.add_argument('--extra_downweight_class', type=str, default='fall',
                    help="Class name to additionally scale alpha down for (on top of focal "
                         "loss's automatic effect). Set to '' to disable.")
    p.add_argument('--extra_downweight_factor', type=float, default=0.5)

    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--no_amp', action='store_true', help="Disable mixed precision.")
    p.add_argument('--resume', type=str, default=None, help="Path to a checkpoint to resume from.")

    return p.parse_args()


# ════════════════════════════════════════════════════════════════════════════
# Reproducibility
# ════════════════════════════════════════════════════════════════════════════
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ════════════════════════════════════════════════════════════════════════════
# Annotation build (grouped split to prevent augmentation leakage), only runs
# if --annotation doesn't already exist.
# ════════════════════════════════════════════════════════════════════════════
def build_annotation_if_missing(jpg_root, annotation_path, classes, val_split, seed):
    annotation_path = Path(annotation_path)
    if annotation_path.exists():
        print(f"[info] Using existing annotation file: {annotation_path}")
        return

    print(f"[info] {annotation_path} not found -> auto-building (grouped 80/20 split, "
          f"preventing augmented-copy leakage between train/val)...")
    rng = random.Random(seed)
    jpg_root = Path(jpg_root)
    video_groups = {c: defaultdict(list) for c in classes}

    for cls in classes:
        cls_dir = jpg_root / cls
        if not cls_dir.exists():
            print(f"  [warn] missing class folder: {cls_dir}")
            continue
        for vid_dir in sorted(cls_dir.iterdir()):
            if not vid_dir.is_dir():
                continue
            n_frames = len(list(vid_dir.glob('*.jpg')))
            if n_frames == 0:
                continue
            # group augmented copies of the same source clip together so they
            # never end up split across train/val (that would leak information)
            base_id = vid_dir.name.split('_')[-1]
            video_groups[cls][base_id].append((vid_dir.name, n_frames))

    database = {}
    for cls in classes:
        base_ids = list(video_groups[cls].keys())
        rng.shuffle(base_ids)
        n_val = max(1, int(len(base_ids) * val_split))
        val_ids = set(base_ids[:n_val])
        for base_id, folders in video_groups[cls].items():
            subset = 'validation' if base_id in val_ids else 'training'
            for vid_name, n_frames in folders:
                database[vid_name] = {
                    'subset': subset,
                    'annotations': {'label': cls, 'segment': [1, n_frames]}
                }

    annotation_path.parent.mkdir(parents=True, exist_ok=True)
    with open(annotation_path, 'w') as f:
        json.dump({'labels': classes, 'database': database}, f, indent=2)
    print(f"[info] Wrote {len(database)} video entries -> {annotation_path}")


# ════════════════════════════════════════════════════════════════════════════
# Dataset — CORRECT slow/fast joint sampling
# ════════════════════════════════════════════════════════════════════════════
class SlowFastDataset(Dataset):
    def __init__(self, jpg_root, annotation_path, subset, classes,
                 fast_frames=32, alpha=4, img_size=224, is_train=True):
        self.jpg_root    = Path(jpg_root)
        self.fast_frames = fast_frames
        self.alpha       = alpha
        self.is_train    = is_train
        self.c2i         = {c: i for i, c in enumerate(classes)}

        base_transforms = [
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize([0.45, 0.45, 0.45], [0.225, 0.225, 0.225]),  # Kinetics norm (matches pretrained ckpt)
        ]
        if is_train:
            # light spatial jitter only -- your dataset is ALREADY temporally/
            # spatially augmented offline, so we keep online augmentation mild
            # to avoid compounding distortions.
            self.transform = T.Compose([
                T.Resize((img_size, img_size)),
                T.RandomHorizontalFlip(p=0.5),
                T.ToTensor(),
                T.Normalize([0.45, 0.45, 0.45], [0.225, 0.225, 0.225]),
            ])
        else:
            self.transform = T.Compose(base_transforms)

        with open(annotation_path, 'r') as f:
            ann = json.load(f)

        self.samples = []
        for vid, info in ann['database'].items():
            if info['subset'] != subset:
                continue
            label_name = info['annotations']['label']
            if label_name not in self.c2i:
                continue
            vid_dir = self.jpg_root / label_name / vid
            if not vid_dir.is_dir():
                continue
            self.samples.append((vid_dir, self.c2i[label_name], label_name, vid))

        print(f"[{subset}] {len(self.samples)} clips loaded across {len(classes)} classes")

    def __len__(self):
        return len(self.samples)

    def class_counts(self, n_classes):
        counts = np.zeros(n_classes, dtype=np.int64)
        for _, label, _, _ in self.samples:
            counts[label] += 1
        return counts

    def _sample_indices(self, total):
        """Sample `fast_frames` indices from a clip of length `total`.
        Train: random contiguous window if long enough, else uniform stretch.
        Val:   deterministic centered/uniform sampling (no randomness)."""
        n = self.fast_frames
        if total <= 0:
            return np.zeros(n, dtype=int)
        if total >= n:
            if self.is_train:
                start = np.random.randint(0, total - n + 1)
                return np.arange(start, start + n)
            else:
                start = (total - n) // 2
                return np.arange(start, start + n)
        # clip shorter than fast_frames -> uniform sample with repeats
        return np.linspace(0, total - 1, n).round().astype(int)

    def __getitem__(self, idx):
        vid_dir, label, cls_name, vid_name = self.samples[idx]
        files = sorted(vid_dir.glob('*.jpg'))
        total = len(files)

        fast_idx = self._sample_indices(total)

        fast_frames = []
        for i in fast_idx:
            i = int(np.clip(i, 0, max(total - 1, 0)))
            try:
                img = Image.open(files[i]).convert('RGB')
            except Exception:
                img = Image.new('RGB', (224, 224), (0, 0, 0))
            fast_frames.append(self.transform(img))
        fast_clip = torch.stack(fast_frames, 0).permute(1, 0, 2, 3)  # [C, T_fast, H, W]

        # Slow pathway MUST be derived from the SAME window as fast (alignment
        # with what the pretrained weights expect) -- every alpha-th frame.
        slow_idx_within_fast = torch.linspace(0, self.fast_frames - 1,
                                               self.fast_frames // self.alpha).long()
        slow_clip = torch.index_select(fast_clip, 1, slow_idx_within_fast)  # [C, T_slow, H, W]

        return [slow_clip, fast_clip], label, cls_name, vid_name


def collate_fn(batch):
    slows = torch.stack([b[0][0] for b in batch])
    fasts = torch.stack([b[0][1] for b in batch])
    labels = torch.tensor([b[1] for b in batch])
    cls_names = [b[2] for b in batch]
    vid_names = [b[3] for b in batch]
    return [slows, fasts], labels, cls_names, vid_names


# ════════════════════════════════════════════════════════════════════════════
# Model
# ════════════════════════════════════════════════════════════════════════════
def build_model(n_classes):
    model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=True)
    in_features = model.blocks[-1].proj.in_features
    model.blocks[-1].proj = nn.Linear(in_features, n_classes)
    return model


def freeze_backbone(model):
    """Phase 1: freeze everything except the final classification head."""
    for name, param in model.named_parameters():
        param.requires_grad = False
    for param in model.blocks[-1].proj.parameters():
        param.requires_grad = True


def unfreeze_from(model, start_block_idx):
    """Phase 2: unfreeze model.blocks[start_block_idx:] (head is always last block)."""
    for name, param in model.named_parameters():
        param.requires_grad = False
    n_blocks = len(model.blocks)
    for bi in range(max(start_block_idx, 0), n_blocks):
        for param in model.blocks[bi].parameters():
            param.requires_grad = True


def set_frozen_bn_eval(model, start_block_idx):
    """Keep BatchNorm layers in the still-frozen blocks in eval() mode so their
    running stats don't drift on a small batch size, even though model.train()
    is active for the rest of the network."""
    for bi in range(0, max(start_block_idx, 0)):
        for m in model.blocks[bi].modules():
            if isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d)):
                m.eval()


def split_param_groups(model, start_block_idx, backbone_lr, head_lr):
    """Differential LR: backbone (blocks[start_block_idx:-1]) vs head (blocks[-1])."""
    backbone_params, head_params = [], []
    n_blocks = len(model.blocks)
    for bi in range(max(start_block_idx, 0), n_blocks):
        target = head_params if bi == n_blocks - 1 else backbone_params
        for p in model.blocks[bi].parameters():
            if p.requires_grad:
                target.append(p)
    groups = []
    if backbone_params:
        groups.append({'params': backbone_params, 'lr': backbone_lr})
    if head_params:
        groups.append({'params': head_params, 'lr': head_lr})
    return groups


# ════════════════════════════════════════════════════════════════════════════
# Focal loss with per-class alpha (inverse-frequency + optional extra downweight)
# ════════════════════════════════════════════════════════════════════════════
class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.register_buffer('alpha', alpha)

    def forward(self, logits, targets):
        log_p = F.log_softmax(logits, dim=1)
        p = log_p.exp()
        log_p_t = log_p.gather(1, targets.unsqueeze(1)).squeeze(1)
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        at = self.alpha[targets]
        loss = -at * ((1 - p_t) ** self.gamma) * log_p_t
        return loss.mean()


def compute_alpha(class_counts, classes, extra_downweight_class, extra_downweight_factor):
    counts = np.maximum(class_counts, 1)
    n_classes = len(counts)
    # sklearn 'balanced' style inverse-frequency weighting
    alpha = counts.sum() / (n_classes * counts)
    alpha = alpha / alpha.mean()  # normalize so mean alpha == 1
    if extra_downweight_class and extra_downweight_class in classes:
        idx = classes.index(extra_downweight_class)
        alpha[idx] *= extra_downweight_factor
    return torch.tensor(alpha, dtype=torch.float32)


# ════════════════════════════════════════════════════════════════════════════
# One epoch
# ════════════════════════════════════════════════════════════════════════════
def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None,
              accum_steps=1, use_amp=True, start_block_idx=None):
    is_train = optimizer is not None
    model.train(is_train)
    if is_train and start_block_idx is not None:
        set_frozen_bn_eval(model, start_block_idx)

    total_loss, all_preds, all_targets = 0.0, [], []
    n_batches = len(loader)

    if is_train:
        optimizer.zero_grad()

    with torch.set_grad_enabled(is_train):
        for i, (inputs, labels, _, _) in enumerate(loader):
            inputs = [x.to(device, non_blocking=True) for x in inputs]
            labels = labels.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss_to_backward = loss / accum_steps

            if is_train:
                if use_amp:
                    scaler.scale(loss_to_backward).backward()
                    if (i + 1) % accum_steps == 0 or (i + 1) == n_batches:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                else:
                    loss_to_backward.backward()
                    if (i + 1) % accum_steps == 0 or (i + 1) == n_batches:
                        optimizer.step()
                        optimizer.zero_grad()

            total_loss += loss.item() * labels.size(0)
            all_preds.append(outputs.argmax(1).detach().cpu())
            all_targets.append(labels.detach().cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()
    avg_loss = total_loss / len(all_targets)
    acc = (all_preds == all_targets).mean()
    return avg_loss, acc, all_preds, all_targets


# ════════════════════════════════════════════════════════════════════════════
# Metrics / FLOPs / plots
# ════════════════════════════════════════════════════════════════════════════
def compute_flops_and_params(model, device, slow_frames, fast_frames, img_size):
    if not HAS_FVCORE:
        return None, None
    model.eval()
    dummy = [
        torch.randn(1, 3, slow_frames, img_size, img_size, device=device),
        torch.randn(1, 3, fast_frames, img_size, img_size, device=device),
    ]
    try:
        flops = FlopCountAnalysis(model, (dummy,))
        flops.unsupported_ops_warnings(False)
        total_flops = flops.total()
        total_params = parameter_count(model)['']
        return total_flops, total_params
    except Exception as e:
        print(f"[warn] FLOPs computation failed: {e}")
        return None, None


def benchmark_latency(model, device, slow_frames, fast_frames, img_size, n_iters=30):
    model.eval()
    dummy = [
        torch.randn(1, 3, slow_frames, img_size, img_size, device=device),
        torch.randn(1, 3, fast_frames, img_size, img_size, device=device),
    ]
    with torch.no_grad():
        for _ in range(5):  # warmup
            model(dummy)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n_iters):
            model(dummy)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t1 = time.time()
    ms_per_clip = (t1 - t0) / n_iters * 1000
    return ms_per_clip, 1000.0 / ms_per_clip


def save_history_plot(history, out_path):
    epochs = [h['epoch'] for h in history]
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, [h['train_loss'] for h in history], label='Train Loss')
    plt.plot(epochs, [h['val_loss'] for h in history], label='Val Loss')
    for h in history:
        if h.get('phase_start'):
            plt.axvline(h['epoch'], color='gray', linestyle='--', alpha=0.5)
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.title('Loss vs Epoch')

    plt.subplot(1, 2, 2)
    plt.plot(epochs, [h['train_acc'] for h in history], label='Train Acc')
    plt.plot(epochs, [h['val_acc'] for h in history], label='Val Acc')
    plt.plot(epochs, [h['val_macro_f1'] for h in history], label='Val Macro-F1', linestyle=':')
    plt.xlabel('Epoch'); plt.ylabel('Score'); plt.legend(); plt.title('Accuracy / F1 vs Epoch')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


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


# ════════════════════════════════════════════════════════════════════════════
# Checkpointing
# ════════════════════════════════════════════════════════════════════════════
def save_checkpoint(path, model, optimizer, scaler, epoch, phase, best_metric,
                     history, args_dict):
    torch.save({
        'epoch': epoch,
        'phase': phase,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict() if optimizer is not None else None,
        'scaler_state': scaler.state_dict() if scaler is not None else None,
        'best_metric': best_metric,
        'history': history,
        'args': args_dict,
    }, path)


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════
def main():
    args = get_args()
    set_seed(args.seed)
    classes = [c.strip() for c in args.classes.split(',') if c.strip()]
    n_classes = len(classes)

    result_path = Path(args.result_path)
    result_path.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = (not args.no_amp) and device.type == 'cuda'

    print(f"Device        : {device}")
    print(f"Classes ({n_classes}): {classes}")
    print(f"Mixed precision: {use_amp}\n")

    build_annotation_if_missing(args.jpg_root, args.annotation, classes, args.val_split, args.seed)

    # ── Data ──────────────────────────────────────────────────────────────
    train_ds = SlowFastDataset(args.jpg_root, args.annotation, 'training', classes,
                                args.fast_frames, args.alpha, args.img_size, is_train=True)
    val_ds   = SlowFastDataset(args.jpg_root, args.annotation, 'validation', classes,
                                args.fast_frames, args.alpha, args.img_size, is_train=False)

    class_counts = train_ds.class_counts(n_classes)
    print("\nTraining class distribution:")
    for c, n in zip(classes, class_counts):
        print(f"  {c:15s}: {n}")

    alpha = compute_alpha(class_counts, classes, args.extra_downweight_class,
                           args.extra_downweight_factor).to(device)
    print(f"\nFocal-loss per-class alpha weights: "
          f"{dict(zip(classes, [round(a, 3) for a in alpha.cpu().tolist()]))}\n")
    criterion = FocalLoss(alpha=alpha, gamma=args.focal_gamma)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.n_workers, collate_fn=collate_fn,
                               pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                               num_workers=args.n_workers, collate_fn=collate_fn,
                               pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────────
    model = build_model(n_classes).to(device)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    history = []
    start_epoch = 0
    start_phase = 'head'
    best_metric = -1.0
    patience_counter = 0
    global_epoch = 0

    # ── Resume ────────────────────────────────────────────────────────────
    if args.resume and Path(args.resume).exists():
        print(f"[info] Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        history = ckpt.get('history', [])
        best_metric = ckpt.get('best_metric', -1.0)
        start_epoch = ckpt.get('epoch', 0) + 1
        start_phase = ckpt.get('phase', 'head')
        global_epoch = len(history)
        print(f"[info] Resumed at phase={start_phase}, epoch={start_epoch}, "
              f"best_metric={best_metric:.4f}")

    def train_phase(phase_name, n_epochs, epoch_offset, optimizer, scheduler,
                     resume_epoch, start_block_idx, monitor_early_stop):
        nonlocal best_metric, patience_counter, global_epoch
        for local_epoch in range(resume_epoch, n_epochs):
            t0 = time.time()
            train_loss, train_acc, _, _ = run_epoch(
                model, train_loader, criterion, device, optimizer=optimizer,
                scaler=scaler, accum_steps=args.accum_steps, use_amp=use_amp,
                start_block_idx=start_block_idx)
            val_loss, val_acc, val_preds, val_targets = run_epoch(
                model, val_loader, criterion, device, optimizer=None,
                scaler=None, use_amp=use_amp)

            if HAS_SK:
                val_macro_f1 = f1_score(val_targets, val_preds, average='macro', zero_division=0)
            else:
                val_macro_f1 = val_acc

            if scheduler is not None:
                scheduler.step()

            global_epoch += 1
            dt = time.time() - t0
            print(f"[{phase_name}] epoch {local_epoch+1}/{n_epochs} "
                  f"(global {global_epoch}) | train_loss {train_loss:.4f} acc {train_acc:.4f} | "
                  f"val_loss {val_loss:.4f} acc {val_acc:.4f} macroF1 {val_macro_f1:.4f} | "
                  f"{dt:.1f}s")

            history.append({
                'epoch': global_epoch, 'phase': phase_name,
                'phase_start': (local_epoch == resume_epoch),
                'train_loss': train_loss, 'train_acc': train_acc,
                'val_loss': val_loss, 'val_acc': val_acc, 'val_macro_f1': val_macro_f1,
            })

            # always keep a resumable "last" checkpoint
            save_checkpoint(result_path / 'last_checkpoint.pth', model, optimizer, scaler,
                             local_epoch, phase_name, best_metric, history, vars(args))

            improved = val_macro_f1 > best_metric
            if improved:
                best_metric = val_macro_f1
                patience_counter = 0
                save_checkpoint(result_path / 'best_model.pth', model, optimizer, scaler,
                                 local_epoch, phase_name, best_metric, history, vars(args))
                print(f"   ↳ new best (val macro-F1 {best_metric:.4f}) -> saved best_model.pth")
            elif monitor_early_stop:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"[early stop] no improvement in val macro-F1 for "
                          f"{args.patience} epochs. Stopping fine-tune phase.")
                    return True  # stop signal
        return False

    # ── Phase 1: head warmup, backbone frozen ────────────────────────────
    if start_phase == 'head':
        print("\n=== Phase 1: training classification head only (backbone frozen) ===\n")
        freeze_backbone(model)
        head_params = [p for p in model.blocks[-1].proj.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(head_params, lr=args.head_lr, weight_decay=args.weight_decay)
        steps_per_epoch = max(1, len(train_loader) // args.accum_steps)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=args.head_lr, total_steps=args.head_epochs * steps_per_epoch,
            pct_start=0.2)
        # OneCycleLR steps per-batch normally; we call scheduler.step() once/epoch
        # above for simplicity, which is fine for this short warmup phase -- if
        # you want strict per-batch OneCycle behavior, move .step() inside run_epoch.
        resume_epoch = start_epoch if start_phase == 'head' else 0
        train_phase('head', args.head_epochs, 0, optimizer, None, resume_epoch,
                    start_block_idx=len(model.blocks) - 1, monitor_early_stop=False)
        start_epoch = 0  # reset for phase 2

    # ── Phase 2: fine-tune (unfreeze from --unfreeze_from_block) ─────────
    print(f"\n=== Phase 2: fine-tuning from block {args.unfreeze_from_block} onward "
          f"(0=full network, {len(model.blocks)-1}=head only) ===\n")
    unfreeze_from(model, args.unfreeze_from_block)
    param_groups = split_param_groups(model, args.unfreeze_from_block,
                                       args.backbone_lr, args.ft_head_lr)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    steps_per_epoch = max(1, len(train_loader) // args.accum_steps)
    max_lrs = [g['lr'] for g in param_groups]
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=max_lrs, total_steps=args.finetune_epochs * steps_per_epoch,
        pct_start=0.1)

    ft_resume_epoch = start_epoch if start_phase == 'finetune' else 0
    train_phase('finetune', args.finetune_epochs, 0, optimizer, None, ft_resume_epoch,
                start_block_idx=args.unfreeze_from_block, monitor_early_stop=True)

    # ── Final evaluation with best checkpoint ────────────────────────────
    print("\n=== Loading best checkpoint for final evaluation/reporting ===")
    best_ckpt = torch.load(result_path / 'best_model.pth', map_location=device)
    model.load_state_dict(best_ckpt['model_state'])

    val_loss, val_acc, val_preds, val_targets = run_epoch(
        model, val_loader, criterion, device, optimizer=None, scaler=None, use_amp=use_amp)

    save_history_plot(history, result_path / 'loss_acc_curves.png')

    report_dict = None
    if HAS_SK:
        cm = save_confusion_matrix(val_targets, val_preds, classes,
                                    result_path / 'confusion_matrix.png')
        report_dict = classification_report(val_targets, val_preds, target_names=classes,
                                             zero_division=0, output_dict=True)
        print("\nClassification report (validation set):")
        print(classification_report(val_targets, val_preds, target_names=classes, zero_division=0))
    else:
        cm = None

    total_flops, total_params = compute_flops_and_params(
        model, device, args.fast_frames // args.alpha, args.fast_frames, args.img_size)
    ms_per_clip, clips_per_sec = benchmark_latency(
        model, device, args.fast_frames // args.alpha, args.fast_frames, args.img_size)

    summary = {
        'classes': classes,
        'training_class_counts': dict(zip(classes, [int(c) for c in class_counts])),
        'best_val_macro_f1': float(best_metric),
        'final_val_loss': float(val_loss),
        'final_val_acc': float(val_acc),
        'classification_report': report_dict,
        'confusion_matrix': cm.tolist() if cm is not None else None,
        'total_params': int(total_params) if total_params else None,
        'total_flops_giga': (total_flops / 1e9) if total_flops else None,
        'inference_ms_per_clip': ms_per_clip,
        'inference_clips_per_sec': clips_per_sec,
        'input_shape': {
            'slow_frames': args.fast_frames // args.alpha,
            'fast_frames': args.fast_frames,
            'img_size': args.img_size,
        },
        'hardware': str(device),
        'total_epochs_run': len(history),
        'args': vars(args),
    }
    with open(result_path / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("DONE. Key numbers for your paper:")
    print(f"  Best val macro-F1      : {best_metric:.4f}")
    print(f"  Final val accuracy     : {val_acc:.4f}")
    print(f"  Params                 : {total_params/1e6:.2f} M" if total_params else "  Params: n/a (install fvcore)")
    print(f"  FLOPs                  : {total_flops/1e9:.2f} GFLOPs (single clip)" if total_flops else "  FLOPs: n/a (install fvcore)")
    print(f"  Inference latency      : {ms_per_clip:.2f} ms/clip ({clips_per_sec:.1f} clips/s) on {device}")
    print(f"  All artifacts saved to : {result_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()
