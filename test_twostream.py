"""
test_twostream_kaggle.py
Tests Two-Stream R3D-18 model on unseen test data.
- 5 classes: fight, fall, unsafeThrow, unsafeClimb, unsafeJump
- noclass prediction when confidence below threshold
- Computes optical flow on-the-fly from RGB frames

Usage (Kaggle cell):
    import sys
    sys.argv = sys.argv[:1]
    exec(open('test_twostream_kaggle.py').read())
"""

import sys
sys.argv = sys.argv[:1]

import os, json, time, traceback
import numpy as np
from PIL import Image
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models.video import r3d_18
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

try:
    import cv2
    HAS_CV2 = True
except:
    HAS_CV2 = False
    print("[WARN] cv2 not available — flow stream will use zeros")

# ── Config ────────────────────────────────────────────────────────────────────
jpg_root       = '/kaggle/working/TEST_DATA_jpg_raw'
checkpoint     = '/kaggle/input/models/aishajalil/two-stream-model/best_model.pth'
result_path    = '/kaggle/working/test_results_twostream'
n_frames       = 16
img_size       = 112
batch_size     = 8
n_workers      = 2
noclass_thresh = 0.4

CLASSES  = ['fight', 'fall', 'unsafeThrow', 'unsafeClimb', 'unsafeJump']
C2I      = {c: i for i, c in enumerate(CLASSES)}
N_CLS    = len(CLASSES)

os.makedirs(result_path, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice          : {device}")
print(f"Checkpoint      : {checkpoint}")
print(f"noclass_thresh  : {noclass_thresh}\n")

# ── Transforms ────────────────────────────────────────────────────────────────
rgb_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

flow_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

# ── Optical Flow ──────────────────────────────────────────────────────────────
def compute_optical_flow(pil_frames):
    flow_pils = []
    prev_gray = None
    for pil_img in pil_frames:
        arr = np.array(pil_img)
        if HAS_CV2:
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None,
                    0.5, 3, 15, 3, 5, 1.2, 0)
                mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
                hsv = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)
                hsv[...,0] = ang * 180 / np.pi / 2
                hsv[...,1] = 255
                hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
                flow_pils.append(Image.fromarray(rgb))
            else:
                flow_pils.append(Image.fromarray(
                    np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)))
            prev_gray = gray
        else:
            flow_pils.append(Image.fromarray(
                np.zeros((img_size, img_size, 3), dtype=np.uint8)))
    return flow_pils

# ── Dataset ───────────────────────────────────────────────────────────────────
class TestDataset(Dataset):
    def __init__(self, jpg_root, n_frames=16):
        self.n_frames = n_frames
        self.samples  = []
        jpg_root = Path(jpg_root)
        for cls in CLASSES:
            cls_dir = jpg_root / cls
            if not cls_dir.exists():
                print(f"  [WARN] Not found: {cls_dir}")
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

    def _load_pil_frames(self, vid_dir):
        files = sorted(
            list(Path(vid_dir).glob("image_*.jpg")) +
            list(Path(vid_dir).glob("img_*.jpg")) +
            list(Path(vid_dir).glob("*.jpg")))
        total = len(files)
        if total == 0: return []
        indices = np.linspace(0, total-1, self.n_frames, dtype=int)
        frames  = []
        for i in indices:
            try:    frames.append(Image.open(files[i]).convert('RGB'))
            except: frames.append(Image.new('RGB', (img_size, img_size)))
        return frames

    def _to_tensor(self, pil_frames, transform):
        if not pil_frames:
            return torch.zeros(3, self.n_frames, img_size, img_size)
        return torch.stack([transform(f) for f in pil_frames], 0).permute(1,0,2,3)

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        vid_dir, label, cls_name, vid_name = self.samples[idx]
        pil_frames  = self._load_pil_frames(vid_dir)
        rgb_clip    = self._to_tensor(pil_frames, rgb_transform)
        flow_frames = compute_optical_flow(pil_frames)
        flow_clip   = self._to_tensor(flow_frames, flow_transform)
        return [rgb_clip, flow_clip], label, cls_name, vid_name

def collate_fn(batch):
    rgb       = torch.stack([b[0][0] for b in batch])
    flow      = torch.stack([b[0][1] for b in batch])
    labels    = torch.tensor([b[1] for b in batch], dtype=torch.long)
    cls_names = [b[2] for b in batch]
    vid_names = [b[3] for b in batch]
    return [rgb, flow], labels, cls_names, vid_names

# ── Model ─────────────────────────────────────────────────────────────────────
class TwoStreamModel(nn.Module):
    def __init__(self, n_classes=5):
        super().__init__()
        rgb_base  = r3d_18(weights=None)
        flow_base = r3d_18(weights=None)
        self.rgb_encoder  = nn.Sequential(*list(rgb_base.children())[:-1])
        self.flow_encoder = nn.Sequential(*list(flow_base.children())[:-1])
        self.classifier   = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.6),
            nn.Linear(512+512, 512), nn.ReLU(inplace=True),
            nn.BatchNorm1d(512), nn.Dropout(0.4),
            nn.Linear(512, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.3), nn.Linear(256, n_classes))

    def forward(self, inputs):
        rf = self.rgb_encoder(inputs[0]).squeeze(-1).squeeze(-1).squeeze(-1)
        ff = self.flow_encoder(inputs[1]).squeeze(-1).squeeze(-1).squeeze(-1)
        return self.classifier(torch.cat([rf, ff], dim=1))

def load_model(ckpt_path):
    ckpt  = torch.load(ckpt_path, map_location=device)
    state = ckpt.get('state_dict', ckpt)
    model = TwoStreamModel(n_classes=N_CLS)
    ns    = {k.replace('module.',''): v for k,v in state.items()}
    model.load_state_dict(ns, strict=True)
    model = model.to(device)
    model.eval()
    ep  = ckpt.get('epoch', '?')
    acc = ckpt.get('val_acc', 0)
    print(f"✅ Two-Stream loaded — epoch {ep}, val acc: {acc*100:.2f}%")
    return model

# ── Inference ─────────────────────────────────────────────────────────────────
def run_inference(model, loader, threshold):
    all_preds, all_labels = [], []
    all_probs, all_names  = [], []
    all_max_conf          = []
    noclass_count         = 0

    with torch.no_grad():
        for i, (inputs, labels, _, vnames) in enumerate(loader):
            inputs  = [x.to(device) for x in inputs]
            outputs = model(inputs)
            probs   = torch.softmax(outputs, dim=1)
            confs, preds = probs.max(dim=1)

            for j in range(len(labels)):
                conf = confs[j].item()
                pred = preds[j].item()
                if conf < threshold:
                    pred = -1
                    noclass_count += 1
                all_preds.append(pred)
                all_labels.append(labels[j].item())
                all_probs.append(probs[j].cpu().numpy())
                all_max_conf.append(conf)
            all_names.extend(vnames)
            print(f"  Batch {i+1}/{len(loader)}", end='\r')

    print(f"\n  noclass: {noclass_count}/{len(all_preds)}")
    return (np.array(all_preds), np.array(all_labels),
            np.array(all_probs), all_names, np.array(all_max_conf))

# ── Plots ─────────────────────────────────────────────────────────────────────
def plot_confusion_matrix(cm, path):
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap='Blues'); plt.colorbar(im, ax=ax)
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
    ax.set_title('Confusion Matrix — Two-Stream R3D-18\n(Unseen Test Data)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  Saved: {path}")

def plot_per_class_metrics(report, path):
    metrics = ['precision', 'recall', 'f1-score']
    x = np.arange(N_CLS); w = 0.25
    colors = ['#3498DB', '#2ECC71', '#E74C3C']
    fig, ax = plt.subplots(figsize=(11, 6))
    for i, (m, c) in enumerate(zip(metrics, colors)):
        ax.bar(x+i*w, [report.get(cls,{}).get(m,0) for cls in CLASSES],
               w, label=m.capitalize(), color=c, alpha=0.85)
    ax.set_xticks(x+w); ax.set_xticklabels(CLASSES, fontsize=10, rotation=15)
    ax.set_ylabel('Score', fontsize=12); ax.set_ylim(0, 1.15)
    ax.axhline(0.8, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_title('Per-Class Metrics — Two-Stream R3D-18',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  Saved: {path}")

def plot_f1_bar(report, acc, path):
    f1s    = [report.get(c,{}).get('f1-score',0) for c in CLASSES]
    colors = ['#2ECC71' if f>=0.8 else '#F39C12' if f>=0.6 else '#E74C3C'
              for f in f1s]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(CLASSES, f1s, color=colors, edgecolor='white')
    ax.axhline(acc, color='#2C3E50', linestyle='--', linewidth=2,
               label=f'Acc (excl. noclass) = {acc:.2%}')
    ax.set_ylim(0, 1.15); ax.set_ylabel('F1-Score', fontsize=12)
    ax.set_title('Per-Class F1 — Two-Stream', fontsize=12, fontweight='bold')
    for bar, v in zip(bars, f1s):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                f'{v:.2f}', ha='center', fontsize=11, fontweight='bold')
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  Saved: {path}")

def plot_confidence(max_confs, preds, labels, path, threshold):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    correct  = preds == labels
    noclass  = preds == -1
    ax.hist(max_confs[correct & ~noclass],  bins=20, alpha=0.7,
            color='#2ECC71', label=f'Correct ({(correct&~noclass).sum()})')
    ax.hist(max_confs[~correct & ~noclass], bins=20, alpha=0.7,
            color='#E74C3C', label=f'Wrong ({(~correct&~noclass).sum()})')
    ax.hist(max_confs[noclass],             bins=20, alpha=0.7,
            color='#95A5A6', label=f'noclass ({noclass.sum()})')
    ax.axvline(threshold, color='black', linestyle='--', linewidth=2,
               label=f'Threshold={threshold}')
    ax.set_xlabel('Max Confidence', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Confidence Distribution', fontweight='bold')
    ax.legend(fontsize=10)
    ax2 = axes[1]
    thresholds   = np.arange(0.1, 0.95, 0.05)
    accs, ncrates = [], []
    for t in thresholds:
        mp = preds.copy(); mp[max_confs < t] = -1
        cls = mp != -1
        accs.append((mp[cls]==labels[cls]).mean()*100 if cls.sum()>0 else 0)
        ncrates.append((~cls).mean()*100)
    ax2.plot(thresholds, accs,    color='#2ECC71', linewidth=2,
             marker='o', markersize=3, label='Accuracy')
    ax2_r = ax2.twinx()
    ax2_r.plot(thresholds, ncrates, color='#E74C3C', linewidth=2,
               marker='s', markersize=3, linestyle='--', label='noclass %')
    ax2.axvline(threshold, color='black', linestyle=':', linewidth=2)
    ax2.set_xlabel('Threshold', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', color='#2ECC71', fontsize=12)
    ax2_r.set_ylabel('noclass Rate (%)', color='#E74C3C', fontsize=12)
    ax2.set_title('Threshold Analysis', fontweight='bold')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  Saved: {path}")

def save_csv(names, labels, preds, probs, max_confs, path):
    with open(path, 'w') as f:
        f.write('video_name,true_label,predicted_label,is_noclass,correct,'
                'max_confidence,' +
                ','.join([f'prob_{c}' for c in CLASSES]) + '\n')
        for n, l, p, pr, conf in zip(names, labels, preds, probs, max_confs):
            pred_cls = 'noclass' if p==-1 else CLASSES[p]
            f.write(f"{n},{CLASSES[l]},{pred_cls},{int(p==-1)},"
                    f"{int(l==p)},{conf:.4f},"
                    + ','.join([f'{x:.4f}' for x in pr]) + '\n')
    print(f"  Saved: {path}")

# ── MAIN ──────────────────────────────────────────────────────────────────────
try:
    ds = TestDataset(jpg_root, n_frames)
    if len(ds) == 0:
        print(f"❌ No samples in {jpg_root}")
    else:
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                            num_workers=n_workers, collate_fn=collate_fn,
                            pin_memory=True)
        model = load_model(checkpoint)

        print(f"\n⏳ Running inference (noclass threshold: {noclass_thresh})...")
        t0 = time.time()
        preds, labels, probs, names, max_confs = run_inference(
            model, loader, noclass_thresh)
        elapsed = time.time() - t0

        classified  = preds != -1
        noclass_cnt = (~classified).sum()
        acc_all     = (preds == labels).mean()
        acc_cls     = ((preds[classified] == labels[classified]).mean()
                       if classified.sum() > 0 else 0.0)

        print(f"\n{'='*55}")
        print(f"  Total videos             : {len(preds)}")
        print(f"  noclass (low conf)       : {noclass_cnt} ({noclass_cnt/len(preds)*100:.1f}%)")
        print(f"  Overall Accuracy         : {acc_all*100:.2f}%")
        print(f"  Accuracy (excl. noclass) : {acc_cls*100:.2f}%")
        print(f"  Inference time           : {elapsed:.1f}s")
        print(f"{'='*55}\n")

        if HAS_SK and classified.sum() > 0:
            report = classification_report(
                labels[classified], preds[classified],
                target_names=CLASSES, labels=list(range(N_CLS)),
                output_dict=True, zero_division=0)
            report_str = classification_report(
                labels[classified], preds[classified],
                target_names=CLASSES, labels=list(range(N_CLS)),
                zero_division=0)
            cm = confusion_matrix(
                labels[classified], preds[classified],
                labels=list(range(N_CLS)))
            print(report_str)

            with open(os.path.join(result_path,
                                   'classification_report.txt'), 'w') as f:
                f.write(f"Model: Two-Stream R3D-18\n"
                        f"Classes: {CLASSES}\n"
                        f"noclass threshold: {noclass_thresh}\n\n"
                        f"Total: {len(preds)}  noclass: {noclass_cnt}\n"
                        f"Overall Acc: {acc_all*100:.2f}%\n"
                        f"Acc (excl. noclass): {acc_cls*100:.2f}%\n\n")
                f.write(report_str)

            with open(os.path.join(result_path,
                                   'confusion_matrix.json'), 'w') as f:
                json.dump({'matrix': cm.tolist(), 'classes': CLASSES,
                           'accuracy_all': float(acc_all),
                           'accuracy_classified': float(acc_cls),
                           'noclass_count': int(noclass_cnt)}, f, indent=2)

            print("📊 Saving graphs...")
            plot_confusion_matrix(
                cm, os.path.join(result_path, '1_confusion_matrix.png'))
            plot_per_class_metrics(
                report, os.path.join(result_path, '2_per_class_metrics.png'))
            plot_f1_bar(
                report, acc_cls,
                os.path.join(result_path, '3_f1_summary.png'))
            plot_confidence(
                max_confs, preds, labels,
                os.path.join(result_path, '4_confidence_threshold.png'),
                noclass_thresh)

        save_csv(names, labels, preds, probs, max_confs,
                 os.path.join(result_path, 'per_video_results.csv'))

        print(f"\n✅ Results saved to: {result_path}")

except Exception as e:
    traceback.print_exc()
