import os
import json
import argparse
import time
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import MultiStepLR

# ── SlowFast from pytorchvideo ────────────────────────────────────────────────
try:
    from pytorchvideo.models import create_slowfast
    HAS_PTV = True
except ImportError:
    HAS_PTV = False
    print("[WARN] pytorchvideo not found. Install: pip install pytorchvideo")

# ── Arguments ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--jpg_root',    type=str, default='/kaggle/working/FYP_DATA_jpg_raw')
parser.add_argument('--annotation',  type=str, default='/kaggle/working/FYP_DATA_jpg_raw/dataset.json')
parser.add_argument('--result_path', type=str, default='/kaggle/working/results_slowfast')
parser.add_argument('--n_epochs',    type=int, default=30)
parser.add_argument('--batch_size',  type=int, default=8)
parser.add_argument('--lr',          type=float, default=0.001)
parser.add_argument('--n_workers',   type=int, default=4)
parser.add_argument('--checkpoint',  type=int, default=2, help='Save model every N epochs')
parser.add_argument('--slow_frames', type=int, default=8)
parser.add_argument('--fast_frames', type=int, default=32)
parser.add_argument('--img_size',    type=int, default=224)
parser.add_argument('--resume_path', type=str, default=None)
args = parser.parse_args()

# ── Configuration ─────────────────────────────────────────────────────────────
CLASSES   = ['fight', 'unsafeClimb', 'unsafeJump', 'unsafeThrow', 'fall']
C2I       = {c: i for i, c in enumerate(CLASSES)}
N_CLASSES = len(CLASSES)
os.makedirs(args.result_path, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ── Dataset ───────────────────────────────────────────────────────────────────
class SlowFastDataset(Dataset):
    def __init__(self, jpg_root, annotation_path, subset,
                 slow_frames=8, fast_frames=32, img_size=224):
        self.jpg_root    = jpg_root
        self.slow_frames = slow_frames
        self.fast_frames = fast_frames
        self.img_size    = img_size

        with open(annotation_path, 'r') as f:
            ann = json.load(f)

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.45, 0.45, 0.45], [0.225, 0.225, 0.225]),
        ])

        self.samples = []
        for vid, info in ann['database'].items():
            if info['subset'] != subset: continue
            lbl = info['annotations']['label']
            idx = C2I.get(lbl, -1)
            if idx < 0: continue
            
            vid_dir = os.path.join(jpg_root, lbl, vid)
            if os.path.isdir(vid_dir):
                self.samples.append((vid_dir, idx))

        print(f"[{subset}] {len(self.samples)} videos loaded")

    def _load_frames(self, vid_dir, n_frames, is_fast=False):
        files = sorted([f for f in os.listdir(vid_dir) if f.endswith('.jpg')])
        total = len(files)
        if total == 0:
            return torch.zeros(3, n_frames, self.img_size, self.img_size)

        if not is_fast:
            # Slow Path: Sparse sampling across entire video
            indices = np.linspace(0, total - 1, n_frames, dtype=int)
        else:
            # Fast Path: Dense temporal sampling
            if total > n_frames:
                start = np.random.randint(0, total - n_frames)
                indices = np.arange(start, start + n_frames)
            else:
                indices = np.linspace(0, total - 1, n_frames, dtype=int)

        frames = []
        for i in indices:
            try:
                img = Image.open(os.path.join(vid_dir, files[i])).convert('RGB')
            except:
                img = Image.new('RGB', (self.img_size, self.img_size))
            frames.append(self.transform(img))

        return torch.stack(frames, 0).permute(1, 0, 2, 3) # [C, T, H, W]

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        vid_dir, label = self.samples[idx]
        slow = self._load_frames(vid_dir, self.slow_frames, is_fast=False)
        fast = self._load_frames(vid_dir, self.fast_frames, is_fast=True)
        return [slow, fast], label

# ── Model Construction ────────────────────────────────────────────────────────
def build_model():
    if HAS_PTV:
        try:
            model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=True)
            in_features = model.blocks[-1].proj.in_features
            model.blocks[-1].proj = nn.Linear(in_features, N_CLASSES)
            print("✅ SlowFast-R50 loaded from torch.hub")
            return model
        except Exception as e:
            print(f"[WARN] torch.hub failed: {e}")

    # Fallback Manual Model
    from torchvision.models.video import r3d_18
    class TwoStreamSlowFast(nn.Module):
        def __init__(self, n_classes):
            super().__init__()
            self.slow = r3d_18(weights=None)
            self.slow.fc = nn.Identity()
            self.fast = r3d_18(weights=None)
            self.fast.fc = nn.Identity()
            self.classifier = nn.Sequential(
                nn.Dropout(0.5), nn.Linear(1024, 512), nn.ReLU(),
                nn.Dropout(0.3), nn.Linear(512, n_classes)
            )
        def forward(self, inputs):
            s, f = self.slow(inputs[0]), self.fast(inputs[1])
            return self.classifier(torch.cat([s, f], dim=1))

    return TwoStreamSlowFast(N_CLASSES)

# ── Training Functions ────────────────────────────────────────────────────────
def run_epoch(model, loader, optimizer, criterion, device, is_train=True):
    model.train() if is_train else model.eval()
    total_loss, correct, total = 0, 0, 0
    
    with torch.set_grad_enabled(is_train):
        for i, (inputs, labels) in enumerate(loader):
            inputs = [x.to(device) for x in inputs]
            labels = labels.to(device)

            if is_train: optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

    return total_loss / len(loader), correct / total

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    train_ds = SlowFastDataset(args.jpg_root, args.annotation, 'training', args.slow_frames, args.fast_frames, args.img_size)
    val_ds   = SlowFastDataset(args.jpg_root, args.annotation, 'validation', args.slow_frames, args.fast_frames, args.img_size)

    def collate_fn(batch):
        return [torch.stack([b[0][0] for b in batch]), torch.stack([b[0][1] for b in batch])], torch.tensor([b[1] for b in batch])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers, collate_fn=collate_fn, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers, collate_fn=collate_fn, pin_memory=True)

    model = build_model().to(device)
    
    # Corrected weights from your class distribution
    weights = torch.tensor([0.913, 0.953, 1.096, 1.196, 0.903]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    scheduler = MultiStepLR(optimizer, milestones=[10, 20, 25], gamma=0.1)

    best_acc = 0.0
    for epoch in range(1, args.n_epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, optimizer, criterion, device, is_train=True)
        vl_loss, vl_acc = run_epoch(model, val_loader, None, criterion, device, is_train=False)
        scheduler.step()

        print(f"Epoch {epoch}/{args.n_epochs} | Train Acc: {tr_acc:.2%} | Val Acc: {vl_acc:.2%}")

        if vl_acc > best_acc:
            best_acc = vl_acc
            torch.save({'state_dict': model.state_dict(), 'val_acc': vl_acc}, os.path.join(args.result_path, 'best_model.pth'))
            print(f"🏆 Saved new best model: {vl_acc:.2%}")

    print(f"✅ Training Complete. Best Val Acc: {best_acc:.2%}")
