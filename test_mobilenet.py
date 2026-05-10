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

# ── Config ────────────────────────────────────────────────────────────────────
jpg_root    = '/kaggle/working/TEST_DATA_jpg_raw'
checkpoint  = '/kaggle/working/results_mobilenet/best_model.pth'
result_path = '/kaggle/working/test_results_mobilenet'
n_frames    = 16
img_size    = 112
batch_size  = 8
n_workers   = 2

# Keep ALL 6 classes to match model training indices exactly
CLASSES = ['fight', 'Normal', 'unsafeClimb', 'unsafeJump', 'unsafeThrow', 'fall']
C2I     = {c: i for i, c in enumerate(CLASSES)}
N_CLS   = len(CLASSES)

os.makedirs(result_path, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice     : {device}")
print(f"Checkpoint : {checkpoint}")
print(f"JPG root   : {jpg_root}\n")

transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

class TestDataset(Dataset):
    def __init__(self, jpg_root, n_frames=16):
        self.n_frames = n_frames
        self.samples  = []
        jpg_root = Path(jpg_root)
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
                if len(frames) == 0: continue
                self.samples.append((str(vid_dir), C2I[cls], cls, vid_dir.name))
        counts = Counter(s[2] for s in self.samples)
        print(f"Total test samples: {len(self.samples)}")
        print("\nPer-class count:")
        for cls in CLASSES:
            print(f"  {cls:15s}: {counts.get(cls, 0)} videos")
        print()

    def _load_clip(self, vid_dir):
        files = sorted(list(Path(vid_dir).glob("image_*.jpg")) +
                       list(Path(vid_dir).glob("img_*.jpg")) +
                       list(Path(vid_dir).glob("*.jpg")))
        total = len(files)
        if total == 0:
            return torch.zeros(3, self.n_frames, img_size, img_size)
        indices = np.linspace(0, total-1, self.n_frames, dtype=int)
        frames  = []
        for i in indices:
            try:    img = Image.open(files[i]).convert('RGB')
            except: img = Image.new('RGB', (img_size, img_size), (0,0,0))
            frames.append(transform(img))
        return torch.stack(frames, 0).permute(1, 0, 2, 3)

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        vid_dir, label, cls_name, vid_name = self.samples[idx]
        return self._load_clip(vid_dir), label, cls_name, vid_name

def collate_fn(batch):
    clips     = torch.stack([b[0] for b in batch])
    labels    = torch.tensor([b[1] for b in batch], dtype=torch.long)
    cls_names = [b[2] for b in batch]
    vid_names = [b[3] for b in batch]
    return clips, labels, cls_names, vid_names

class ConvBnRelu(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel, stride=1, padding=0, groups=1):
        super().__init__(
            nn.Conv3d(in_ch, out_ch, kernel, stride=stride,
                      padding=padding, groups=groups, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU6(inplace=True))

class DepthwiseSeparable3D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.dw = ConvBnRelu(in_ch, in_ch, (3,3,3), (1,stride,stride), (1,1,1), in_ch)
        self.pw = ConvBnRelu(in_ch, out_ch, (1,1,1))
    def forward(self, x): return self.pw(self.dw(x))

class MobileNet3D(nn.Module):
    def __init__(self, n_classes=6, width_mult=1.0):
        super().__init__()
        def ch(c): return max(1, int(c * width_mult))
        self.features = nn.Sequential(
            ConvBnRelu(3, ch(32), (3,3,3), (1,2,2), (1,1,1)),
            DepthwiseSeparable3D(ch(32),   ch(64),   1),
            DepthwiseSeparable3D(ch(64),   ch(128),  2),
            DepthwiseSeparable3D(ch(128),  ch(128),  1),
            DepthwiseSeparable3D(ch(128),  ch(256),  2),
            DepthwiseSeparable3D(ch(256),  ch(256),  1),
            DepthwiseSeparable3D(ch(256),  ch(512),  2),
            DepthwiseSeparable3D(ch(512),  ch(512),  1),
            DepthwiseSeparable3D(ch(512),  ch(512),  1),
            DepthwiseSeparable3D(ch(512),  ch(512),  1),
            DepthwiseSeparable3D(ch(512),  ch(512),  1),
            DepthwiseSeparable3D(ch(512),  ch(512),  1),
            DepthwiseSeparable3D(ch(512),  ch(1024), 2),
            DepthwiseSeparable3D(ch(1024), ch(1024), 1),
        )
        self.pool       = nn.AdaptiveAvgPool3d(1)
        self.dropout    = nn.Dropout(0.5)
        self.classifier = nn.Linear(ch(1024), n_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.classifier(x)

def load_model(ckpt_path):
    ckpt  = torch.load(ckpt_path, map_location=device)
    state = ckpt.get('state_dict', ckpt)
    model = MobileNet3D(n_classes=6, width_mult=1.0)
    new_state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(new_state, strict=True)
    model = model.to(device)
    model.eval()
    ep  = ckpt.get('epoch', '?')
    acc = ckpt.get('val_acc', 0)
    print(f"✅ Model loaded — epoch {ep}, val acc: {acc*100:.2f}%")
    return model

def run_inference(model, loader):
    all_preds, all_labels, all_probs, all_names = [], [], [], []
    with torch.no_grad():
        for i, (clips, labels, _, vnames) in enumerate(loader):
            clips   = clips.to(device)
            outputs = model(clips)
            probs   = torch.softmax(outputs, dim=1)
            preds   = probs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
            all_names.extend(vnames)
            print(f"  Batch {i+1}/{len(loader)}", end='\r')
    print()
    return np.array(all_preds), np.array(all_labels), np.array(all_probs), all_names

def plot_confusion_matrix(cm, path):
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm, cmap='Blues')
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(N_CLS)); ax.set_yticks(range(N_CLS))
    ax.set_xticklabels(CLASSES, rotation=35, ha='right', fontsize=10)
    ax.set_yticklabels(CLASSES, fontsize=10)
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i,j]), ha='center', va='center', fontsize=12,
                    color='white' if cm[i,j] > thresh else 'black')
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_title('Confusion Matrix — MobileNet3D\n(Unseen Test Data)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

def plot_per_class_metrics(report, path):
    metrics = ['precision', 'recall', 'f1-score']
    x = np.arange(N_CLS); w = 0.25
    colors = ['#3498DB', '#2ECC71', '#E74C3C']
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (m, c) in enumerate(zip(metrics, colors)):
        vals = [report.get(cls, {}).get(m, 0) for cls in CLASSES]
        ax.bar(x + i*w, vals, w, label=m.capitalize(), color=c, alpha=0.85)
    ax.set_xticks(x + w)
    ax.set_xticklabels(CLASSES, fontsize=9, rotation=15)
    ax.set_ylabel('Score', fontsize=12); ax.set_ylim(0, 1.15)
    ax.axhline(0.8, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='0.8 target')
    ax.set_title('Per-Class Precision / Recall / F1 — MobileNet3D',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

def plot_f1_bar(report, acc, path):
    f1s    = [report.get(c, {}).get('f1-score', 0) for c in CLASSES]
    colors = ['#2ECC71' if f >= 0.8 else '#F39C12' if f >= 0.6 else '#E74C3C' for f in f1s]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(CLASSES, f1s, color=colors, edgecolor='white')
    ax.axhline(acc, color='#2C3E50', linestyle='--', linewidth=2,
               label=f'Overall Acc = {acc:.2%}')
    ax.set_ylim(0, 1.15); ax.set_ylabel('F1-Score', fontsize=12)
    ax.set_title('Per-Class F1  |  Green≥0.8  Orange≥0.6  Red<0.6',
                 fontsize=12, fontweight='bold')
    for bar, v in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{v:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

def plot_confidence(probs, preds, labels, path):
    n = N_CLS; cols = 3; rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
    axes = axes.flatten()
    fig.suptitle('Confidence Distribution — MobileNet3D', fontsize=13, fontweight='bold')
    for i, cls in enumerate(CLASSES):
        ax = axes[i]; mask = labels == i
        if mask.sum() == 0:
            ax.set_title(cls)
            ax.text(0.5, 0.5, 'No test samples', ha='center', va='center',
                    transform=ax.transAxes)
            continue
        conf = probs[mask, i]; correct = preds[mask] == i
        ax.hist(conf[correct],  bins=10, alpha=0.7, color='#2ECC71',
                label=f'Correct ({correct.sum()})')
        ax.hist(conf[~correct], bins=10, alpha=0.7, color='#E74C3C',
                label=f'Wrong ({(~correct).sum()})')
        ax.set_title(cls, fontsize=11, fontweight='bold')
        ax.set_xlabel('Confidence'); ax.set_ylabel('Count')
        ax.legend(fontsize=8)
    for j in range(N_CLS, len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

def save_csv(names, labels, preds, probs, path):
    with open(path, 'w') as f:
        f.write('video_name,true_label,predicted_label,correct,' +
                ','.join([f'prob_{c}' for c in CLASSES]) + '\n')
        for n, l, p, pr in zip(names, labels, preds, probs):
            f.write(f"{n},{CLASSES[l]},{CLASSES[p]},{int(l==p)},"
                    + ','.join([f'{x:.4f}' for x in pr]) + '\n')
    print(f"  Saved: {path}")

# ── MAIN ──────────────────────────────────────────────────────────────────────
try:
    ds = TestDataset(jpg_root, n_frames)
    if len(ds) == 0:
        print("❌ No test samples found!")
        print(f"   Check: {jpg_root}")
        print(f"   Expected subfolders: {CLASSES}")
    else:
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                            num_workers=n_workers, collate_fn=collate_fn,
                            pin_memory=True)
        model = load_model(checkpoint)

        print("\n⏳ Running inference...")
        t0 = time.time()
        preds, labels, probs, names = run_inference(model, loader)
        elapsed = time.time() - t0

        acc = (preds == labels).mean()
        print(f"\n{'='*55}")
        print(f"  Overall Accuracy : {acc*100:.2f}%")
        print(f"  Total videos     : {len(preds)}")
        print(f"  Correct          : {(preds==labels).sum()}")
        print(f"  Wrong            : {(preds!=labels).sum()}")
        print(f"  Inference time   : {elapsed:.1f}s")
        print(f"{'='*55}\n")

        if HAS_SK:
            report     = classification_report(labels, preds, target_names=CLASSES,
                                               output_dict=True, zero_division=0)
            report_str = classification_report(labels, preds, target_names=CLASSES,
                                               zero_division=0)
            cm = confusion_matrix(labels, preds, labels=list(range(N_CLS)))
            print(report_str)

            with open(os.path.join(result_path, 'classification_report.txt'), 'w') as f:
                f.write(f"Model: MobileNet3D\nOverall Accuracy: {acc*100:.2f}%\n"
                        f"Total Videos: {len(preds)}\nInference Time: {elapsed:.1f}s\n\n")
                f.write(report_str)

            with open(os.path.join(result_path, 'confusion_matrix.json'), 'w') as f:
                json.dump({'matrix': cm.tolist(), 'classes': CLASSES,
                           'accuracy': float(acc)}, f, indent=2)

            print("📊 Saving graphs...")
            plot_confusion_matrix(cm, os.path.join(result_path, '1_confusion_matrix.png'))
            plot_per_class_metrics(report, os.path.join(result_path, '2_per_class_metrics.png'))
            plot_f1_bar(report, acc, os.path.join(result_path, '3_f1_summary.png'))
            plot_confidence(probs, preds, labels, os.path.join(result_path, '4_confidence_distribution.png'))

        save_csv(names, labels, preds, probs, os.path.join(result_path, 'per_video_results.csv'))
        print(f"\n✅ All results saved to: {result_path}")

except Exception as e:
    traceback.print_exc()
