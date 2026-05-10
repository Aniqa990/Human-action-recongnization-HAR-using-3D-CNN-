"""
train_slowfast.py
Trains SlowFast-R50 on your FYP child safety dataset.
SlowFast has two pathways:
  - Slow (4fps):  understands SCENE CONTEXT — height, environment, objects
  - Fast (32fps): understands MOTION        — speed, energy, trajectory

This directly addresses safe vs unsafe confusion because:
  unsafe_jump: slow path sees HIGH table/window → dangerous context
  safe_jump:   slow path sees LOW bed/chair     → safe context

Usage:
    pip install pytorchvideo
    python train_slowfast.py \
        --jpg_root    "G:/My Drive/Segmented_FYP_DATA_jpg" \
        --annotation  "G:/My Drive/Segmented_FYP_DATA_jpg/dataset.json" \
        --result_path "G:/My Drive/Segmented_FYP_DATA_jpg/results_slowfast"
"""






# ... [Keep SlowFastDataset and build_model function code the same as your original] ...

# ── Training loop Update ──────────────────────────────────────────────────────
# [Inside the Main block, update the saving logic]
# ... 
    
# ...









import os, json, argparse, time
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
  


# ── args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--jpg_root',    type=str, default='/kaggle/working/FYP_DATA_jpg_raw')
parser.add_argument('--annotation',  type=str, default='/kaggle/working/FYP_DATA_jpg_raw/dataset.json')
parser.add_argument('--result_path', type=str, default='/kaggle/working/results_slowfast')
parser.add_argument('--n_epochs',    type=int, default=30)
parser.add_argument('--batch_size',  type=int, default=8)
parser.add_argument('--lr',          type=float, default=0.001)
parser.add_argument('--n_workers',   type=int, default=2)
# NEW: Added checkpoint argument to fix your error
parser.add_argument('--checkpoint',  type=int, default=5, help='Save model every N epochs')
# SlowFast params
parser.add_argument('--slow_frames', type=int, default=8)
parser.add_argument('--fast_frames', type=int, default=32)
parser.add_argument('--img_size',    type=int, default=224)
parser.add_argument('--resume_path', type=str, default=None)
args = parser.parse_args()



# UPDATED: Classes to match your 5 folders (Removed 'Normal', added 'fall')
CLASSES   = ['fight', 'unsafeClimb', 'unsafeJump', 'unsafeThrow', 'fall']
C2I       = {c: i for i, c in enumerate(CLASSES)}
N_CLASSES = len(CLASSES)
os.makedirs(args.result_path, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ── Dataset ───────────────────────────────────────────────────────────────────
class SlowFastDataset(Dataset):
    """
    Returns TWO clips per video:
      slow_clip: [3, slow_frames, H, W]  — uniformly sampled (low fps feel)
      fast_clip: [3, fast_frames, H, W]  — densely sampled   (high fps feel)
    """
    def __init__(self, jpg_root, annotation_path, subset,
                 slow_frames=8, fast_frames=32, img_size=112):
        self.jpg_root    = jpg_root
        self.slow_frames = slow_frames
        self.fast_frames = fast_frames

        with open(annotation_path, 'r') as f:
            ann = json.load(f)

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.45, 0.45, 0.45],
                                 [0.225, 0.225, 0.225]),
        ])

        self.samples = []
        for vid, info in ann['database'].items():
            if info['subset'] != subset: continue
            lbl = info['annotations']['label']
            idx = C2I.get(lbl, -1)
            if idx < 0: continue
            # find video dir — try both with and without class subdir
            for candidate in [
                os.path.join(jpg_root, lbl, vid),
                os.path.join(jpg_root, vid),
            ]:
                if os.path.isdir(candidate):
                    self.samples.append((candidate, idx))
                    break

        print(f"[{subset}] {len(self.samples)} videos loaded")

    def _load_frames(self, vid_dir, n_frames):
        files = sorted([f for f in os.listdir(vid_dir) if f.endswith('.jpg')])
        total = len(files)
        if total == 0:
            return torch.zeros(3, n_frames, args.img_size, args.img_size)

        indices = np.linspace(0, total-1, n_frames, dtype=int)
        frames  = []
        for i in indices:
            try:
                img = Image.open(os.path.join(vid_dir, files[i])).convert('RGB')
            except:
                img = Image.new('RGB', (args.img_size, args.img_size))
            frames.append(self.transform(img))

        return torch.stack(frames, 0).permute(1, 0, 2, 3)  # [3, T, H, W]

    def __len__(self):  return len(self.samples)

    def __getitem__(self, idx):
        vid_dir, label = self.samples[idx]
        slow = self._load_frames(vid_dir, self.slow_frames)
        fast = self._load_frames(vid_dir, self.fast_frames)
        return [slow, fast], label


# ── Model ─────────────────────────────────────────────────────────────────────
def build_model():
    if HAS_PTV:
        try:
            model = torch.hub.load('facebookresearch/pytorchvideo',
                                   'slowfast_r50', pretrained=True)
            # replace head
            in_features = model.blocks[-1].proj.in_features
            model.blocks[-1].proj = nn.Linear(in_features, N_CLASSES)
            print("✅ SlowFast-R50 loaded from torch.hub (pretrained Kinetics)")
            return model
        except Exception as e:
            print(f"[WARN] torch.hub failed: {e}")

    # fallback: manual SlowFast-like model using two R(2+1)D streams
    print("Building manual two-stream model as SlowFast fallback...")
    from torchvision.models.video import r3d_18

    class TwoStreamSlowFast(nn.Module):
        def __init__(self, n_classes):
            super().__init__()
            # Slow stream — low fps, more spatial detail
            self.slow = r3d_18(weights=None)
            self.slow.fc = nn.Identity()

            # Fast stream — high fps, more temporal detail
            self.fast = r3d_18(weights=None)
            self.fast.fc = nn.Identity()

            # Fusion + classifier
            self.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(512 + 512, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, n_classes)
            )

        def forward(self, inputs):
            slow_clip, fast_clip = inputs
            # pool temporal dim to fixed size
            slow_feat = self.slow(slow_clip)   # [B, 512]
            fast_feat = self.fast(fast_clip)   # [B, 512]
            fused     = torch.cat([slow_feat, fast_feat], dim=1)
            return self.classifier(fused)

    model = TwoStreamSlowFast(N_CLASSES)
    print("✅ Two-stream SlowFast fallback model built")
    return model


# ── Training loop ─────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, epoch):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for i, (inputs, labels) in enumerate(loader):
        if isinstance(inputs, list):
            inputs = [x.to(device) for x in inputs]
        else:
            inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
        total_loss += loss.item()

        if (i+1) % 10 == 0:
            print(f"  Epoch {epoch} [{i+1}/{len(loader)}]  "
                  f"Loss: {loss.item():.4f}  "
                  f"Acc: {correct/total*100:.1f}%")

    return total_loss / len(loader), correct / total


def val_epoch(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            if isinstance(inputs, list):
                inputs = [x.to(device) for x in inputs]
            else:
                inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss    = criterion(outputs, labels)
            preds   = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
            total_loss += loss.item()

    return total_loss / len(loader), correct / total


def save_log(path, epoch, loss, acc, lr):
    header = not os.path.exists(path)
    with open(path, 'a') as f:
        if header: f.write("epoch\tloss\tacc\tlr\n")
        f.write(f"{epoch}\t{loss:.6f}\t{acc:.6f}\t{lr}\n")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # datasets
    train_ds = SlowFastDataset(args.jpg_root, args.annotation, 'training',
                                args.slow_frames, args.fast_frames, args.img_size)
    val_ds   = SlowFastDataset(args.jpg_root, args.annotation, 'validation',
                                args.slow_frames, args.fast_frames, args.img_size)

    def collate(batch):
        slow  = torch.stack([b[0][0] for b in batch])
        fast  = torch.stack([b[0][1] for b in batch])
        labels = torch.tensor([b[1] for b in batch])
        return [slow, fast], labels

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=args.n_workers,
                              collate_fn=collate, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.n_workers,
                              collate_fn=collate, pin_memory=True)

    # model
    model = build_model().to(device)

    # weighted loss for class imbalance
    # Normal is dominant so give it lower weight
    weights = torch.tensor([1.61, 0.53, 1.53, 1.17, 1.03]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                 momentum=0.9, weight_decay=1e-4)
    scheduler = MultiStepLR(optimizer, milestones=[10, 20, 25], gamma=0.1)

    start_epoch = 1
    if args.resume_path and os.path.exists(args.resume_path):
        ckpt = torch.load(args.resume_path, map_location=device)
        model.load_state_dict(ckpt['state_dict'])
        start_epoch = ckpt['epoch'] + 1
        print(f"✅ Resumed from epoch {ckpt['epoch']}")

    train_log = os.path.join(args.result_path, 'train.txt')
    val_log   = os.path.join(args.result_path, 'val.txt')

    best_val_acc = 0.0
    print(f"\n🚀 Starting SlowFast training for {args.n_epochs} epochs\n")

    for epoch in range(start_epoch, args.n_epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, epoch)
        vl_loss, vl_acc = val_epoch(model, val_loader, criterion)
        scheduler.step()

        lr = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0

        # Keep your progress prints!
        print(f"\nEpoch {epoch}/{args.n_epochs}  "
              f"Train Loss: {tr_loss:.4f}  Train Acc: {tr_acc*100:.2f}%  |  "
              f"Val Loss: {vl_loss:.4f}  Val Acc: {vl_acc*100:.2f}%  "
              f"[{elapsed:.0f}s]\n")

        # Keep your text logs for plotting later
        save_log(train_log, epoch, tr_loss, tr_acc, lr)
        save_log(val_log,   epoch, vl_loss, vl_acc, lr)

        # UPDATED: Dynamic saving frequency based on your command line argument
        if epoch % args.checkpoint == 0:
            ckpt_path = os.path.join(args.result_path, f'save_{epoch}.pth')
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict()}, ckpt_path)
            print(f"💾 Checkpoint saved: {ckpt_path}")

        # IMPORTANT: Keep the best model save logic
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            best_path = os.path.join(args.result_path, 'best_model.pth')
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(),
                        'val_acc': vl_acc}, best_path)
            print(f"🏆 New best val acc: {vl_acc*100:.2f}% → saved best_model.pth")

    print(f"\n✅ Training complete. Best Val Acc: {best_val_acc*100:.2f}%")
    print(f"   Results saved to: {args.result_path}")
