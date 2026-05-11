"""
test_slowfast_kaggle.py
Tests SlowFast / Two-Stream model on unseen test data.
- 5 classes: fight, fall, unsafeThrow, unsafeClimb, unsafeJump
- "noclass" prediction when model confidence is below threshold
- Reads JPG frames directly from folder — NO annotation file needed

Folder structure:
    TEST_DATA_jpg_raw/
        fight/
            video1/
                image_00001.jpg
        fall/
        unsafeThrow/
        unsafeClimb/
        unsafeJump/

Usage (Kaggle cell):
    import sys
    sys.argv = sys.argv[:1]
    exec(open('test_slowfast_kaggle.py').read())
"""

import sys
sys.argv = sys.argv[:1]  # Fix Kaggle/Colab argparse conflict

import os, json, time, traceback
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

try:
    from sklearn.metrics import confusion_matrix, classification_report
    HAS_SK = True
except:
    HAS_SK = False
    print("[WARN] pip install scikit-learn")

# ── Config — change these ─────────────────────────────────────────────────────
jpg_root        = '/kaggle/working/TEST_DATA_jpg_raw'
checkpoint      = '/kaggle/working/results_slowfast/best_model.pth'
result_path     = '/kaggle/working/test_results_slowfast'
slow_frames     = 8
fast_frames     = 32
img_size        = 224
batch_size      = 4
n_workers       = 2
noclass_thresh  = 0.5   # if max confidence < this → predict "noclass"

# ── Classes ───────────────────────────────────────────────────────────────────
# 5 classes — Normal removed, fall added
CLASSES   = ['fight', 'fall', 'unsafeThrow', 'unsafeClimb', 'unsafeJump']
C2I       = {c: i for i, c in enumerate(CLASSES)}
N_CLS     = len(CLASSES)

# noclass is NOT a real class — it's a label added at test time only
DISPLAY_CLASSES = CLASSES + ['noclass']  # used for CSV only

os.makedirs(result_path, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice          : {device}")
print(f"Checkpoint      : {checkpoint}")
print(f"JPG root        : {jpg_root}")
print(f"noclass_thresh  : {noclass_thresh} (confidence below this → noclass)\n")

# ── Transforms ────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.45, 0.45, 0.45],
                         [0.225, 0.225, 0.225]),
])

# ── Dataset ───────────────────────────────────────────────────────────────────
class TestDataset(Dataset):
    """
    Reads directly from folder structure.
    No annotation JSON needed.
    """
    def __init__(self, jpg_root, slow_frames=8, fast_frames=32):
        self.slow_frames = slow_frames
        self.fast_frames = fast_frames
        self.samples     = []
        jpg_root         = Path(jpg_root)

        for cls in CLASSES:
            cls_dir = jpg_root / cls
            if not cls_dir.exists():
                print(f"  [WARN] Not found: {cls_dir} — skipping")
                continue
            for vid_dir in sorted(cls_dir.iterdir()):
                if not vid_dir.is_dir(): continue
                frames = (list(vid_dir.glob("image_*.jpg")) +
                          list(vid_dir.glob("img_*.jpg")) +
                          list(vid_dir.glob("*.jpg")))
                if len(frames) == 0:
                    print(f"  [SKIP] No frames: {vid_dir.name}")
                    continue
                self.samples.append((str(vid_dir), C2I[cls], cls, vid_dir.name))

        counts = Counter(s[2] for s in self.samples)
        print(f"Total test samples: {len(self.samples)}")
        print("\nPer-class count:")
        for cls in CLASSES:
            print(f"  {cls:15s}: {counts.get(cls, 0)} videos")
        print()

    def _load_frames(self, vid_dir, n):
        files = sorted(list(Path(vid_dir).glob("image_*.jpg")) +
                       list(Path(vid_dir).glob("img_*.jpg")) +
                       list(Path(vid_dir).glob("*.jpg")))
        total = len(files)
        if total == 0:
            return torch.zeros(3, n, img_size, img_size)
        indices = np.linspace(0, total-1, n, dtype=int)
        frames  = []
        for i in indices:
            try:    img = Image.open(files[i]).convert('RGB')
            except: img = Image.new('RGB', (img_size, img_size), (0,0,0))
            frames.append(transform(img))
        return torch.stack(frames, 0).permute(1, 0, 2, 3)  # [3, T, H, W]

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        vid_dir, label, cls_name, vid_name = self.samples[idx]
        slow = self._load_frames(vid_dir, self.slow_frames)
        fast = self._load_frames(vid_dir, self.fast_frames)
        return [slow, fast], label, cls_name, vid_name

def collate_fn(batch):
    slow      = torch.stack([b[0][0] for b in batch])
    fast      = torch.stack([b[0][1] for b in batch])
    labels    = torch.tensor([b[1] for b in batch], dtype=torch.long)
    cls_names = [b[2] for b in batch]
    vid_names = [b[3] for b in batch]
    return [slow, fast], labels, cls_names, vid_names

# ── Model ─────────────────────────────────────────────────────────────────────
class TwoStreamModel(nn.Module):
    """Two-Stream R3D-18 (SlowFast fallback used during training)"""
    def __init__(self, n_classes=5):
        super().__init__()
        from torchvision.models.video import r3d_18
        slow = r3d_18(weights=None)
        fast = r3d_18(weights=None)
        self.slow_encoder = nn.Sequential(*list(slow.children())[:-1])
        self.fast_encoder = nn.Sequential(*list(fast.children())[:-1])
        self.classifier   = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.5),
            nn.Linear(512+512, 512), nn.ReLU(inplace=True),
            nn.Dropout(0.3), nn.Linear(512, n_classes))

    def forward(self, inputs):
        sf = self.slow_encoder(inputs[0]).squeeze(-1).squeeze(-1).squeeze(-1)
        ff = self.fast_encoder(inputs[1]).squeeze(-1).squeeze(-1).squeeze(-1)
        return self.classifier(torch.cat([sf, ff], dim=1))

def load_model(ckpt_path):
    ckpt       = torch.load(ckpt_path, map_location=device)
    state      = ckpt.get('state_dict', ckpt)
    model_type = ckpt.get('model_type', ckpt.get('model', 'unknown'))
    print(f"Checkpoint model_type: '{model_type}'")

    model = None

    # Try SlowFast hub
    try:
        m = torch.hub.load('facebookresearch/pytorchvideo:main',
                           'slowfast_r50', pretrained=False)
        in_feat = m.blocks[-1].proj.in_features
        m.blocks[-1].proj = nn.Linear(in_feat, N_CLS)
        ns = {k.replace('module.',''): v for k,v in state.items()}
        m.load_state_dict(ns, strict=False)
        model = m
        model_type = 'slowfast_hub'
        print("✅ Loaded as SlowFast-R50")
    except Exception as e:
        print(f"   SlowFast hub failed: {e}")

    # Try Two-Stream fallback
    if model is None:
        try:
            m = TwoStreamModel(n_classes=N_CLS)
            ns = {k.replace('module.',''): v for k,v in state.items()}
            m.load_state_dict(ns, strict=True)
            model = m
            model_type = 'two_stream'
            print("✅ Loaded as Two-Stream R3D-18")
        except Exception as e:
            print(f"   Two-Stream failed: {e}")

    if model is None:
        raise RuntimeError("❌ Could not load model!")

    model = model.to(device)
    model.eval()
    ep  = ckpt.get('epoch', '?')
    acc = ckpt.get('val_acc', ckpt.get('best_val_acc', 0))
    print(f"   Epoch: {ep}  |  Val Acc: {acc*100:.2f}%\n")
    return model, model_type

# ── Inference with noclass ─────────────────────────────────────────────────────
def run_inference(model, loader, threshold):
    """
    Returns predictions with noclass support.
    If max confidence < threshold → predicted label = -1 (noclass)
    """
    all_preds        = []  # predicted class index (-1 = noclass)
    all_labels       = []  # true class index
    all_probs        = []  # full probability array
    all_names        = []  # video names
    all_max_conf     = []  # max confidence per video
    noclass_count    = 0

    with torch.no_grad():
        for i, (inputs, labels, _, vnames) in enumerate(loader):
            inputs = [x.to(device) for x in inputs]
            outputs = model(inputs)
            probs   = torch.softmax(outputs, dim=1)
            confs, preds = probs.max(dim=1)

            for j in range(len(labels)):
                conf = confs[j].item()
                pred = preds[j].item()

                if conf < threshold:
                    pred = -1  # noclass
                    noclass_count += 1

                all_preds.append(pred)
                all_labels.append(labels[j].item())
                all_probs.append(probs[j].cpu().numpy())
                all_max_conf.append(conf)
            all_names.extend(vnames)
            print(f"  Batch {i+1}/{len(loader)}", end='\r')

    print(f"\n  noclass predictions: {noclass_count}/{len(all_preds)}")
    return (np.array(all_preds), np.array(all_labels),
            np.array(all_probs), all_names, np.array(all_max_conf))

# ── Plots ─────────────────────────────────────────────────────────────────────
def plot_confusion_matrix(cm, classes, path, title='SlowFast'):
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap='Blues')
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=35, ha='right', fontsize=10)
    ax.set_yticklabels(classes, fontsize=10)
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i,j]), ha='center', va='center',
                    fontsize=12, color='white' if cm[i,j] > thresh else 'black')
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_title(f'Confusion Matrix — {title}\n(Unseen Test Data)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

def plot_per_class_metrics(report, classes, path):
    metrics = ['precision', 'recall', 'f1-score']
    x = np.arange(len(classes)); w = 0.25
    colors = ['#3498DB', '#2ECC71', '#E74C3C']
    fig, ax = plt.subplots(figsize=(11, 6))
    for i, (m, c) in enumerate(zip(metrics, colors)):
        vals = [report.get(cls, {}).get(m, 0) for cls in classes]
        ax.bar(x + i*w, vals, w, label=m.capitalize(), color=c, alpha=0.85)
    ax.set_xticks(x + w)
    ax.set_xticklabels(classes, fontsize=10, rotation=15)
    ax.set_ylabel('Score', fontsize=12); ax.set_ylim(0, 1.15)
    ax.axhline(0.8, color='gray', linestyle='--', linewidth=1,
               alpha=0.5, label='0.8 target')
    ax.set_title('Per-Class Precision / Recall / F1 — SlowFast',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

def plot_f1_bar(report, classes, acc, path):
    f1s    = [report.get(c, {}).get('f1-score', 0) for c in classes]
    colors = ['#2ECC71' if f >= 0.8 else '#F39C12' if f >= 0.6
              else '#E74C3C' for f in f1s]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(classes, f1s, color=colors, edgecolor='white')
    ax.axhline(acc, color='#2C3E50', linestyle='--', linewidth=2,
               label=f'Overall Acc = {acc:.2%}')
    ax.set_ylim(0, 1.15); ax.set_ylabel('F1-Score', fontsize=12)
    ax.set_title('Per-Class F1  |  Green≥0.8  Orange≥0.6  Red<0.6',
                 fontsize=12, fontweight='bold')
    for bar, v in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{v:.2f}', ha='center', va='bottom',
                fontsize=11, fontweight='bold')
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

def plot_confidence_distribution(max_confs, preds, labels, path, threshold):
    """Shows confidence distribution — helps validate noclass threshold."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: overall confidence histogram
    ax = axes[0]
    correct = preds == labels
    noclass = preds == -1
    ax.hist(max_confs[correct & ~noclass],  bins=20, alpha=0.7,
            color='#2ECC71', label=f'Correct ({(correct & ~noclass).sum()})')
    ax.hist(max_confs[~correct & ~noclass], bins=20, alpha=0.7,
            color='#E74C3C', label=f'Wrong ({(~correct & ~noclass).sum()})')
    ax.hist(max_confs[noclass],             bins=20, alpha=0.7,
            color='#95A5A6', label=f'noclass ({noclass.sum()})')
    ax.axvline(threshold, color='black', linestyle='--',
               linewidth=2, label=f'Threshold={threshold}')
    ax.set_xlabel('Max Confidence', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Confidence Distribution (All Videos)', fontweight='bold')
    ax.legend(fontsize=10)

    # Right: per-class confidence box plot
    ax2 = axes[1]
    data = []
    for i, cls in enumerate(CLASSES):
        mask = (labels == i) & (preds != -1)
        if mask.sum() > 0:
            data.append(max_confs[mask])
        else:
            data.append([])
    bp = ax2.boxplot([d if len(d) > 0 else [0] for d in data],
                     labels=CLASSES, patch_artist=True)
    colors = ['#3498DB', '#E74C3C', '#2ECC71', '#F39C12', '#9B59B6']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax2.axhline(threshold, color='black', linestyle='--',
                linewidth=2, label=f'Threshold={threshold}')
    ax2.set_ylabel('Confidence', fontsize=12)
    ax2.set_title('Per-Class Confidence Distribution', fontweight='bold')
    ax2.legend(fontsize=10)
    plt.xticks(rotation=15)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

def plot_noclass_analysis(preds, labels, max_confs, path, threshold):
    """Shows how noclass threshold affects accuracy."""
    thresholds = np.arange(0.1, 0.95, 0.05)
    accs, noclass_rates = [], []

    for t in thresholds:
        masked_preds = preds.copy()
        masked_preds[max_confs < t] = -1
        # accuracy only on classified (non-noclass) videos
        classified = masked_preds != -1
        if classified.sum() > 0:
            acc = (masked_preds[classified] == labels[classified]).mean()
        else:
            acc = 0.0
        accs.append(acc * 100)
        noclass_rates.append((~classified).mean() * 100)

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()
    ax1.plot(thresholds, accs, color='#2ECC71', linewidth=2,
             marker='o', markersize=4, label='Accuracy on classified')
    ax2.plot(thresholds, noclass_rates, color='#E74C3C', linewidth=2,
             marker='s', markersize=4, linestyle='--', label='noclass rate %')
    ax1.axvline(threshold, color='black', linestyle=':',
                linewidth=2, label=f'Current threshold={threshold}')
    ax1.set_xlabel('noclass Threshold', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', color='#2ECC71', fontsize=12)
    ax2.set_ylabel('noclass Rate (%)', color='#E74C3C', fontsize=12)
    ax1.set_title('noclass Threshold Analysis\n'
                  '(Higher threshold = more noclass, higher accuracy on rest)',
                  fontsize=12, fontweight='bold')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1+lines2, labels1+labels2, fontsize=10)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

def save_csv(names, labels, preds, probs, max_confs, path):
    with open(path, 'w') as f:
        f.write('video_name,true_label,predicted_label,is_noclass,correct,'
                'max_confidence,' +
                ','.join([f'prob_{c}' for c in CLASSES]) + '\n')
        for n, l, p, pr, conf in zip(names, labels, preds, probs, max_confs):
            true_cls = CLASSES[l]
            pred_cls = 'noclass' if p == -1 else CLASSES[p]
            is_nc    = int(p == -1)
            correct  = int(l == p)  # noclass = wrong
            f.write(f"{n},{true_cls},{pred_cls},{is_nc},{correct},{conf:.4f},"
                    + ','.join([f'{x:.4f}' for x in pr]) + '\n')
    print(f"  Saved: {path}")

# ── MAIN ──────────────────────────────────────────────────────────────────────
try:
    # Dataset
    ds = TestDataset(jpg_root, slow_frames, fast_frames)
    if len(ds) == 0:
        print("❌ No test samples found!")
        print(f"   Check: {jpg_root}")
        print(f"   Expected subfolders: {CLASSES}")
    else:
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                            num_workers=n_workers, collate_fn=collate_fn,
                            pin_memory=True)

        # Model
        model, model_type = load_model(checkpoint)

        # Inference
        print(f"\n⏳ Running inference (noclass threshold: {noclass_thresh})...")
        t0 = time.time()
        preds, labels, probs, names, max_confs = run_inference(
            model, loader, noclass_thresh)
        elapsed = time.time() - t0

        # Metrics — exclude noclass from accuracy
        classified  = preds != -1
        noclass_cnt = (~classified).sum()
        acc_all     = (preds == labels).mean()
        acc_cls     = ((preds[classified] == labels[classified]).mean()
                       if classified.sum() > 0 else 0.0)

        print(f"\n{'='*55}")
        print(f"  Total videos         : {len(preds)}")
        print(f"  noclass (low conf)   : {noclass_cnt} "
              f"({noclass_cnt/len(preds)*100:.1f}%)")
        print(f"  Classified videos    : {classified.sum()}")
        print(f"  Overall Accuracy     : {acc_all*100:.2f}% (incl. noclass as wrong)")
        print(f"  Accuracy (excl. noclass): {acc_cls*100:.2f}%")
        print(f"  Inference time       : {elapsed:.1f}s")
        print(f"{'='*55}\n")

        if HAS_SK and classified.sum() > 0:
            # Only evaluate on classified videos
            report = classification_report(
                labels[classified], preds[classified],
                target_names=CLASSES,
                labels=list(range(N_CLS)),
                output_dict=True, zero_division=0)
            report_str = classification_report(
                labels[classified], preds[classified],
                target_names=CLASSES,
                labels=list(range(N_CLS)),
                zero_division=0)
            cm = confusion_matrix(
                labels[classified], preds[classified],
                labels=list(range(N_CLS)))

            print("Classification Report (excluding noclass videos):")
            print(report_str)

            # Save text report
            with open(os.path.join(result_path,
                                   'classification_report.txt'), 'w') as f:
                f.write(f"Model: SlowFast / Two-Stream\n")
                f.write(f"Classes: {CLASSES}\n")
                f.write(f"noclass threshold: {noclass_thresh}\n\n")
                f.write(f"Total videos        : {len(preds)}\n")
                f.write(f"noclass predictions : {noclass_cnt} "
                        f"({noclass_cnt/len(preds)*100:.1f}%)\n")
                f.write(f"Overall Accuracy    : {acc_all*100:.2f}%\n")
                f.write(f"Accuracy (excl. noclass): {acc_cls*100:.2f}%\n\n")
                f.write(report_str)

            # Save confusion matrix JSON
            with open(os.path.join(result_path,
                                   'confusion_matrix.json'), 'w') as f:
                json.dump({
                    'matrix': cm.tolist(),
                    'classes': CLASSES,
                    'accuracy_all': float(acc_all),
                    'accuracy_classified': float(acc_cls),
                    'noclass_count': int(noclass_cnt),
                    'noclass_threshold': noclass_thresh
                }, f, indent=2)

            # Plots
            print("📊 Saving graphs...")
            plot_confusion_matrix(
                cm, CLASSES,
                os.path.join(result_path, '1_confusion_matrix.png'),
                title=model_type)
            plot_per_class_metrics(
                report, CLASSES,
                os.path.join(result_path, '2_per_class_metrics.png'))
            plot_f1_bar(
                report, CLASSES, acc_cls,
                os.path.join(result_path, '3_f1_summary.png'))
            plot_confidence_distribution(
                max_confs, preds, labels,
                os.path.join(result_path, '4_confidence_distribution.png'),
                noclass_thresh)
            plot_noclass_analysis(
                preds, labels, max_confs,
                os.path.join(result_path, '5_noclass_threshold_analysis.png'),
                noclass_thresh)

        save_csv(names, labels, preds, probs, max_confs,
                 os.path.join(result_path, 'per_video_results.csv'))

        print(f"\n✅ All results saved to: {result_path}")
        print(f"\nFiles generated:")
        print(f"  1_confusion_matrix.png")
        print(f"  2_per_class_metrics.png")
        print(f"  3_f1_summary.png")
        print(f"  4_confidence_distribution.png")
        print(f"  5_noclass_threshold_analysis.png  ← helps pick best threshold")
        print(f"  classification_report.txt")
        print(f"  confusion_matrix.json")
        print(f"  per_video_results.csv")

except Exception as e:
    traceback.print_exc()
