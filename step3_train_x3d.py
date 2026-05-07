"""
step3_train_x3d.py
Trains X3D-S on FYP child safety dataset.

WHY X3D FOR YOUR PROBLEM:
  - X3D progressively expands only dimensions that matter:
      * Temporal depth    → captures full action arc
      * Frame rate        → captures motion speed
      * Spatial resolution → captures scene context (height, objects)
      * Network width     → captures feature richness
  - Much faster than SlowFast (~10-15 mins/epoch)
  - Better accuracy than MobileNet3D
  - Pretrained on Kinetics-400
  - Single stream → no dual-pathway complexity

COMPARISON:
  SlowFast : best accuracy, very slow (3hrs/epoch)  ❌
  X3D-S    : great accuracy, fast (10-15min/epoch)  ✅ ← this script
  MobileNet: good accuracy, fastest (5-10min/epoch) ✅

Usage:
    pip install pytorchvideo

    python step3_train_x3d.py \
        --jpg_root    "G:/My Drive/FYP_DATA_jpg_raw" \
        --annotation  "G:/My Drive/FYP_DATA_jpg_raw/dataset.json" \
        --result_path "G:/My Drive/FYP_DATA_jpg_raw/results_x3d" \
        --n_epochs 30 \
        --batch_size 16

    # Resume:
    python step3_train_x3d.py ... \
        --resume_path "G:/My Drive/.../results_x3d/save_5.pth"
"""

import os, json, argparse, time
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import MultiStepLR

parser = argparse.ArgumentParser()
parser.add_argument('--jpg_root',    type=str,
                    default='G:/My Drive/FYP_DATA_jpg_raw')
parser.add_argument('--annotation',  type=str,
                    default='G:/My Drive/FYP_DATA_jpg_raw/dataset.json')
parser.add_argument('--result_path', type=str,
                    default='G:/My Drive/FYP_DATA_jpg_raw/results_x3d')
parser.add_argument('--n_epochs',    type=int,   default=30)
parser.add_argument('--batch_size',  type=int,   default=16)
parser.add_argument('--lr',          type=float, default=0.01)
parser.add_argument('--n_workers',   type=int,   default=2)
parser.add_argument('--n_frames',    type=int,   default=16,
                    help='Frames per clip (X3D-S uses 13, but 16 works fine)')
parser.add_argument('--img_size',    type=int,   default=160,
                    help='X3D-S default is 160x160')
parser.add_argument('--x3d_model',   type=str,   default='x3d_s',
                    choices=['x3d_xs', 'x3d_s', 'x3d_m', 'x3d_l'],
                    help='x3d_xs=fastest, x3d_s=balanced, x3d_m=better, x3d_l=best')
parser.add_argument('--checkpoint',  type=int,   default=1,
                    help='Save checkpoint every N epochs')
parser.add_argument('--resume_path', type=str,   default=None)
args = parser.parse_args()

CLASSES   = ['fight', 'Normal', 'unsafeClimb', 'unsafeJump', 'unsafeThrow', 'fall']
C2I       = {c: i for i, c in enumerate(CLASSES)}
N_CLASSES = len(CLASSES)
os.makedirs(args.result_path, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"\nDevice     : {device}")
print(f"X3D Model  : {args.x3d_model}")
print(f"Frames     : {args.n_frames}")
print(f"Image size : {args.img_size}x{args.img_size}\n")

# ── Augmentation ──────────────────────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((args.img_size + 20, args.img_size + 20)),
    transforms.RandomCrop(args.img_size),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.45, 0.45, 0.45],
                         [0.225, 0.225, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((args.img_size, args.img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.45, 0.45, 0.45],
                         [0.225, 0.225, 0.225]),
])

# ── Dataset ───────────────────────────────────────────────────────────────────
class VideoDataset(Dataset):
    def __init__(self, jpg_root, annotation_path, subset,
                 transform, n_frames=16):
        self.jpg_root  = jpg_root
        self.n_frames  = n_frames
        self.transform = transform

        with open(annotation_path, 'r') as f:
            ann = json.load(f)

        self.samples = []
        for vid, info in ann['database'].items():
            if info['subset'] != subset:
                continue
            lbl = info['annotations']['label']
            idx = C2I.get(lbl, -1)
            if idx < 0:
                continue
            for cand in [
                os.path.join(jpg_root, lbl, vid),
                os.path.join(jpg_root, vid),
            ]:
                if os.path.isdir(cand):
                    self.samples.append((cand, idx))
                    break

        print(f"[{subset:10s}] {len(self.samples)} videos")

    def _load_clip(self, vid_dir):
        files = sorted([f for f in os.listdir(vid_dir)
                        if f.endswith('.jpg')])
        total = len(files)
        if total == 0:
            return torch.zeros(3, self.n_frames,
                               args.img_size, args.img_size)

        indices = np.linspace(0, total - 1, self.n_frames, dtype=int)
        frames  = []
        for i in indices:
            try:
                img = Image.open(
                    os.path.join(vid_dir, files[i])).convert('RGB')
            except:
                img = Image.new('RGB',
                                (args.img_size, args.img_size), (0,0,0))
            frames.append(self.transform(img))

        return torch.stack(frames, 0).permute(1, 0, 2, 3)  # [3, T, H, W]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        vid_dir, label = self.samples[idx]
        return self._load_clip(vid_dir), label

# ── Model ─────────────────────────────────────────────────────────────────────
def build_model():
    # Try 1: torch.hub X3D (pretrained Kinetics-400)
    try:
        model = torch.hub.load(
            'facebookresearch/pytorchvideo',
            args.x3d_model,
            pretrained=True
        )
        # Replace final projection head for our classes
        # X3D head: model.blocks[-1].proj
        in_features = model.blocks[-1].proj.in_features
        model.blocks[-1].proj = nn.Linear(in_features, N_CLASSES)
        print(f"✅ {args.x3d_model.upper()} loaded from torch.hub "
              f"(pretrained Kinetics-400)")
        return model, 'x3d_hub'
    except Exception as e:
        print(f"[INFO] torch.hub failed: {e}")

    # Try 2: pytorchvideo direct
    try:
        from pytorchvideo.models.x3d import create_x3d
        model = create_x3d(
            input_clip_length=args.n_frames,
            input_crop_size=args.img_size,
            model_num_class=N_CLASSES,
        )
        print(f"✅ X3D built via pytorchvideo.models")
        return model, 'x3d_ptv'
    except Exception as e:
        print(f"[INFO] pytorchvideo direct failed: {e}")

    # Fallback: R3D-18 (torchvision built-in, always available)
    print("[WARN] X3D not available — using R3D-18 fallback")
    print("       Install pytorchvideo: pip install pytorchvideo")
    from torchvision.models.video import r3d_18, R3D_18_Weights
    model = r3d_18(weights=R3D_18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, N_CLASSES)
    print("✅ R3D-18 fallback loaded (pretrained Kinetics-400)")
    return model, 'r3d18_fallback'

# ── Logging ───────────────────────────────────────────────────────────────────
def save_log(path, epoch, loss, acc, lr):
    header = not os.path.exists(path)
    with open(path, 'a') as f:
        if header:
            f.write("epoch\tloss\tacc\tlr\n")
        f.write(f"{epoch}\t{loss:.6f}\t{acc:.6f}\t{lr:.8f}\n")

def save_batch_log(path, epoch, batch, iteration, loss, acc, lr):
    header = not os.path.exists(path)
    with open(path, 'a') as f:
        if header:
            f.write("epoch\tbatch\titer\tloss\tacc\tlr\n")
        f.write(f"{epoch}\t{batch}\t{iteration}\t"
                f"{loss:.6f}\t{acc:.6f}\t{lr:.8f}\n")

# ── Train / Val ───────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, epoch, batch_log, lr):
    model.train()
    total_loss, correct, total = 0, 0, 0
    offset = (epoch - 1) * len(loader)

    for i, (clips, labels) in enumerate(loader):
        clips  = clips.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(clips)
        loss    = criterion(outputs, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        preds     = outputs.argmax(dim=1)
        b_correct = (preds == labels).sum().item()
        b_acc     = b_correct / labels.size(0)
        correct  += b_correct
        total    += labels.size(0)
        total_loss += loss.item()

        save_batch_log(batch_log, epoch, i+1, offset+i+1,
                       loss.item(), b_acc, lr)

        if (i+1) % 10 == 0 or (i+1) == len(loader):
            print(f"  [{epoch}][{i+1}/{len(loader)}]  "
                  f"Loss: {loss.item():.4f}  "
                  f"Acc: {correct/total*100:.1f}%", end='\r')
    print()
    return total_loss / len(loader), correct / total


def val_epoch(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    class_correct = [0] * N_CLASSES
    class_total   = [0] * N_CLASSES

    with torch.no_grad():
        for clips, labels in loader:
            clips  = clips.to(device)
            labels = labels.to(device)

            outputs = model(clips)
            loss    = criterion(outputs, labels)
            preds   = outputs.argmax(dim=1)

            correct    += (preds == labels).sum().item()
            total      += labels.size(0)
            total_loss += loss.item()

            for lbl, pred in zip(labels.cpu(), preds.cpu()):
                class_total[lbl] += 1
                if lbl == pred:
                    class_correct[lbl] += 1

    per_class = {}
    for i in range(N_CLASSES):
        acc_pct = 100 * class_correct[i] / max(1, class_total[i])
        per_class[CLASSES[i]] = (
            f"{class_correct[i]}/{class_total[i]} ({acc_pct:.1f}%)"
        )
    return total_loss / len(loader), correct / total, per_class

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Loading datasets...")
    train_ds = VideoDataset(args.jpg_root, args.annotation,
                            'training', train_transform, args.n_frames)
    val_ds   = VideoDataset(args.jpg_root, args.annotation,
                            'validation', val_transform, args.n_frames)

    if len(train_ds) == 0:
        print("❌ No training samples! Check --jpg_root and --annotation")
        exit(1)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.n_workers, pin_memory=True,
        persistent_workers=args.n_workers > 0)

    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.n_workers, pin_memory=True,
        persistent_workers=args.n_workers > 0)

    print("\nBuilding model...")
    model, model_type = build_model()
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {total_params/1e6:.2f}M")

    # weighted loss
    # fight=1.61, Normal=0.53, unsafeClimb=1.53, unsafeJump=1.17,
    # unsafeThrow=1.03, fall=1.50
    weights   = torch.tensor([1.61, 0.53, 1.53, 1.17, 1.03, 1.50]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr,
        momentum=0.9, weight_decay=1e-4, nesterov=True)
    scheduler = MultiStepLR(
        optimizer, milestones=[10, 20, 25], gamma=0.1)

    train_log = os.path.join(args.result_path, 'train.txt')
    val_log   = os.path.join(args.result_path, 'val.txt')
    batch_log = os.path.join(args.result_path, 'train_batch.txt')

    # resume
    start_epoch  = 1
    best_val_acc = 0.0
    if args.resume_path and os.path.exists(args.resume_path):
        ckpt = torch.load(args.resume_path, map_location=device)
        
        # Check model type compatibility
        ckpt_model_type = ckpt.get('model_type', 'unknown')
        if ckpt_model_type != model_type:
            print(f"⚠️  MODEL TYPE MISMATCH!")
            print(f"    Checkpoint was saved with: {ckpt_model_type}")
            print(f"    Current model type:        {model_type}")
            print(f"    Loading with strict=False (keys will be skipped)\n")
        
        # Try strict loading first, fall back to non-strict
        try:
            model.load_state_dict(ckpt['state_dict'], strict=True)
        except RuntimeError as e:
            if "Missing key" in str(e) or "Unexpected key" in str(e):
                print(f"⚠️  Architecture mismatch detected, loading with strict=False")
                model.load_state_dict(ckpt['state_dict'], strict=False)
            else:
                raise
        
        if 'optimizer' in ckpt and ckpt_model_type == model_type:
            try:
                optimizer.load_state_dict(ckpt['optimizer'])
            except Exception as e:
                print(f"⚠️  Could not load optimizer state: {e}")
        
        if 'scheduler' in ckpt and ckpt_model_type == model_type:
            try:
                scheduler.load_state_dict(ckpt['scheduler'])
            except Exception as e:
                print(f"⚠️  Could not load scheduler state: {e}")
        
        start_epoch  = ckpt.get('epoch', 1) + 1
        best_val_acc = ckpt.get('best_val_acc', 0.0)
        print(f"✅ Resumed from epoch {start_epoch-1} "
              f"(best val: {best_val_acc*100:.2f}%)")

    # save opts
    import json as _json
    with open(os.path.join(args.result_path, 'opts.json'), 'w') as f:
        _json.dump(vars(args), f, indent=2)

    print(f"\n{'='*60}")
    print(f" Model      : {model_type}")
    print(f" Params     : {total_params/1e6:.2f}M")
    print(f" Classes    : {N_CLASSES} → {CLASSES}")
    print(f" Train vids : {len(train_ds)}")
    print(f" Val vids   : {len(val_ds)}")
    print(f" Epochs     : {args.n_epochs}")
    print(f" Batch size : {args.batch_size}")
    print(f" Frames     : {args.n_frames}")
    print(f" Image size : {args.img_size}x{args.img_size}")
    print(f" Device     : {device}")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, args.n_epochs + 1):
        t0 = time.time()
        lr = optimizer.param_groups[0]['lr']

        tr_loss, tr_acc = train_epoch(
            model, train_loader, optimizer,
            criterion, epoch, batch_log, lr)

        vl_loss, vl_acc, per_cls = val_epoch(
            model, val_loader, criterion)

        scheduler.step()
        elapsed = time.time() - t0

        print(f"Epoch {epoch:3d}/{args.n_epochs}  "
              f"| Train Loss: {tr_loss:.4f}  Acc: {tr_acc*100:.2f}%"
              f"  | Val Loss: {vl_loss:.4f}  Acc: {vl_acc*100:.2f}%"
              f"  | {elapsed:.0f}s")

        print("  Per-class val:")
        for cls, stat in per_cls.items():
            pct = float(stat.split('(')[1].replace('%)', ''))
            marker = "✅" if pct >= 70 else ("⚠️" if pct >= 50 else "❌")
            print(f"    {marker} {cls:15s}: {stat}")
        print()

        save_log(train_log, epoch, tr_loss, tr_acc, lr)
        save_log(val_log,   epoch, vl_loss, vl_acc, lr)

        # checkpoint
        if epoch % args.checkpoint == 0:
            # Save in result_path directory
            p_result = os.path.join(args.result_path, f'save_{epoch}.pth')
            print(f"Saving checkpoint to: {p_result}")
            try:
                torch.save({
                    'epoch':        epoch,
                    'model_type':   model_type,
                    'state_dict':   model.state_dict(),
                    'optimizer':    optimizer.state_dict(),
                    'scheduler':    scheduler.state_dict(),
                    'best_val_acc': best_val_acc,
                }, p_result)
                print(f"  💾 Checkpoint saved in result_path: save_{epoch}.pth")
            except Exception as e:
                print(f"Failed to save checkpoint in result_path: {e}")

            # Save in x3d_result folder in project root
            x3d_root_dir = os.path.join(os.getcwd(), 'x3d_result')
            os.makedirs(x3d_root_dir, exist_ok=True)
            p_root = os.path.join(x3d_root_dir, f'save_{epoch}.pth')
            print(f"Saving checkpoint to project root x3d_result: {p_root}")
            try:
                torch.save({
                    'epoch':        epoch,
                    'model_type':   model_type,
                    'state_dict':   model.state_dict(),
                    'optimizer':    optimizer.state_dict(),
                    'scheduler':    scheduler.state_dict(),
                    'best_val_acc': best_val_acc,
                }, p_root)
                print(f"  💾 Checkpoint saved in project root x3d_result: save_{epoch}.pth")
            except Exception as e:
                print(f"Failed to save checkpoint in project root x3d_result: {e}")

        # best model
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            p = os.path.join(args.result_path, 'best_model.pth')
            torch.save({
                'epoch':      epoch,
                'model_type': model_type,
                'state_dict': model.state_dict(),
                'val_acc':    vl_acc,
            }, p)
            print(f"  🏆 Best val acc: {vl_acc*100:.2f}% → best_model.pth\n")

    print(f"\n✅ Training complete!")
    print(f"   Best Val Acc : {best_val_acc*100:.2f}%")
    print(f"   Results      : {args.result_path}")
