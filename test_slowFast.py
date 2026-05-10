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
# UPDATED: Paths and sizes to match your SlowFast Training
jpg_root    = '/kaggle/working/TEST_DATA_jpg_raw'
checkpoint  = '/kaggle/working/results_slowfast/best_model.pth'
result_path = '/kaggle/working/test_results_slowfast'
slow_frames = 8
fast_frames = 32
img_size    = 224
batch_size  = 8
n_workers   = 2

CLASSES = ['fight', 'unsafeClimb', 'unsafeJump', 'unsafeThrow', 'fall']
C2I     = {c: i for i, c in enumerate(CLASSES)}
N_CLS   = len(CLASSES)

os.makedirs(result_path, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# UPDATED: Normalization to match train_slowfast.py
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.45, 0.45, 0.45], [0.225, 0.225, 0.225]),
])

# ── Dataset ───────────────────────────────────────────────────────────────────
class SlowFastTestDataset(Dataset):
    def __init__(self, jpg_root):
        self.samples  = []
        root_path = Path(jpg_root)
        for cls in CLASSES:
            cls_dir = root_path / cls
            if not cls_dir.exists(): continue
            for vid_dir in sorted(cls_dir.iterdir()):
                if not vid_dir.is_dir(): continue
                self.samples.append((str(vid_dir), C2I[cls], cls, vid_dir.name))
        
        print(f"Total test samples: {len(self.samples)}")

    def _load_frames(self, vid_dir, n_frames):
        files = sorted(list(Path(vid_dir).glob("*.jpg")))
        total = len(files)
        if total == 0: return torch.zeros(3, n_frames, img_size, img_size)
        indices = np.linspace(0, total-1, n_frames, dtype=int)
        frames  = []
        for i in indices:
            img = Image.open(files[i]).convert('RGB')
            frames.append(transform(img))
        return torch.stack(frames, 0).permute(1, 0, 2, 3)

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        vid_dir, label, _, vname = self.samples[idx]
        slow = self._load_frames(vid_dir, slow_frames)
        fast = self._load_frames(vid_dir, fast_frames)
        return [slow, fast], label, vname

def collate_fn(batch):
    slow = torch.stack([b[0][0] for b in batch])
    fast = torch.stack([b[0][1] for b in batch])
    labels = torch.tensor([b[1] for b in batch])
    vnames = [b[2] for b in batch]
    return [slow, fast], labels, vnames

# ── Model Loader ──────────────────────────────────────────────────────────────
def load_slowfast_model(ckpt_path):
    # This assumes you used the PytorchVideo build from your training script
    import torch.hub
    try:
        model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=False)
        in_features = model.blocks[-1].proj.in_features
        model.blocks[-1].proj = nn.Linear(in_features, N_CLS)
    except:
        # Fallback to the manual two-stream model if Hub fails
        from torchvision.models.video import r3d_18
        class FallbackModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.slow = r3d_18(); self.slow.fc = nn.Identity()
                self.fast = r3d_18(); self.fast.fc = nn.Identity()
                self.classifier = nn.Sequential(nn.Linear(1024, 256), nn.ReLU(), nn.Linear(256, N_CLS))
            def forward(self, x):
                s, f = x
                return self.classifier(torch.cat([self.slow(s), self.fast(f)], dim=1))
        model = FallbackModel()

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['state_dict'])
    model = model.to(device).eval()
    print(f"✅ SlowFast Model Loaded from {ckpt_path}")
    return model
# ── MAIN INFERENCE & VISUALIZATION ──────────────────────────────────────────
if __name__ == '__main__':
    ds = SlowFastTestDataset(jpg_root)
    if len(ds) == 0:
        print("❌ No test samples found! Check your paths.")
    else:
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, 
                            num_workers=n_workers, collate_fn=collate_fn)
        model = load_slowfast_model(checkpoint)

        all_preds, all_labels, all_probs = [], [], []
        
        print("⏳ Running Inference...")
        with torch.no_grad():
            for inputs, labels, _ in loader:
                inputs = [x.to(device) for x in inputs]
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
                all_labels.extend(labels.numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        # 1. Print Standard Report
        print("\n" + "="*30)
        print("TEST RESULTS SUMMARY")
        print("="*30)
        report = classification_report(all_labels, all_preds, target_names=CLASSES, output_dict=True)
        print(classification_report(all_labels, all_preds, target_names=CLASSES))

        # 2. Plot Confusion Matrix
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('SlowFast Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(len(CLASSES))
        plt.xticks(tick_marks, CLASSES, rotation=45)
        plt.yticks(tick_marks, CLASSES)

        # Add text annotations to CM
        thresh = cm.max() / 2.
        for i, j in np.ndindex(cm.shape):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(result_path, 'confusion_matrix.png'))
        print(f"✅ Saved: confusion_matrix.png")

        # 3. Plot Per-Class Accuracy (Bar Chart)
        accuracies = [report[cls]['precision'] for cls in CLASSES]
        plt.figure(figsize=(10, 6))
        plt.bar(CLASSES, accuracies, color='skyblue', edgecolor='navy')
        plt.axhline(y=np.mean(accuracies), color='red', linestyle='--', label=f'Avg: {np.mean(accuracies):.2f}')
        plt.title('Per-Class Precision Score')
        plt.ylabel('Precision')
        plt.ylim(0, 1.1)
        plt.legend()
        plt.savefig(os.path.join(result_path, 'precision_bars.png'))
        print(f"✅ Saved: precision_bars.png")

        # 4. Save Raw Results to CSV for deep-dive
        import pandas as pd
        df = pd.DataFrame({
            'True': [CLASSES[l] for l in all_labels],
            'Pred': [CLASSES[p] for p in all_preds],
            'Correct': all_labels == all_preds
        })
        df.to_csv(os.path.join(result_path, 'test_predictions.csv'), index=False)
        print(f"✅ Saved: test_predictions.csv")

        print(f"\n📊 All visualization files are ready in: {result_path}")
