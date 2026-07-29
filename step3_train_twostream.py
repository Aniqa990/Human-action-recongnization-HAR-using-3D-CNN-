"""
step3_train_twostream.py
Trains Two-Stream R3D-18 model on FYP child safety dataset.

TWO STREAMS:
  Stream 1 (RGB):         captures appearance, scene context, height
  Stream 2 (Optical Flow): captures motion energy, speed, trajectory

WHY TWO-STREAM FOR YOUR PROBLEM:
  safe_jump   → low optical flow magnitude (gentle motion)
  unsafe_jump → HIGH optical flow magnitude (fast, energetic)
  fight       → very chaotic, high flow in multiple directions
  fall        → sudden large flow downward

This directly addresses safe/unsafe confusion because flow
captures motion ENERGY which differs between safe and unsafe.

Usage:
    python step3_train_twostream.py \
        --jpg_root    "G:/My Drive/FYP_DATA_jpg_raw" \
        --annotation  "G:/My Drive/FYP_DATA_jpg_raw/dataset_fixed.json" \
        --result_path "G:/My Drive/FYP_DATA_jpg_raw/results_twostream" \
        --n_epochs 50 \
        --batch_size 16 \
        --lr 0.001

    # Resume:
    python step3_train_twostream.py ... \
        --resume_path ".../results_twostream/save_10.pth"

Kaggle:
    !python step3_train_twostream.py \
        --jpg_root    /kaggle/working/FYP_DATA_jpg_raw \
        --annotation  /kaggle/working/FYP_DATA_jpg_raw/dataset_fixed.json \
        --result_path /kaggle/working/results_twostream \
        --n_epochs 50 \
        --batch_size 16 \
        --lr 0.001
"""

import os, json, argparse, time
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models.video import r3d_18, R3D_18_Weights
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import MultiStepLR

parser = argparse.ArgumentParser()
parser.add_argument('--jpg_root',    type=str,
                    default='G:/My Drive/FYP_DATA_jpg_raw')
parser.add_argument('--annotation',  type=str,
                    default='G:/My Drive/FYP_DATA_jpg_raw/dataset_fixed.json')
parser.add_argument('--result_path', type=str,
                    default='G:/My Drive/FYP_DATA_jpg_raw/results_twostream')
parser.add_argument('--n_epochs',    type=int,   default=50)
parser.add_argument('--batch_size',  type=int,   default=16)
parser.add_argument('--lr',          type=float, default=0.001)
parser.add_argument('--n_workers',   type=int,   default=2)
parser.add_argument('--n_frames',    type=int,   default=16)
parser.add_argument('--img_size',    type=int,   default=112)
parser.add_argument('--checkpoint',  type=int,   default=1)
parser.add_argument('--resume_path', type=str,   default=None)
args = parser.parse_args()

CLASSES   = ['fight', 'fall', 'unsafeThrow', 'unsafeClimb', 'unsafeJump']
C2I       = {c: i for i, c in enumerate(CLASSES)}
N_CLASSES = len(CLASSES)
os.makedirs(args.result_path, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"\nDevice     : {device}")
print(f"Frames     : {args.n_frames}")
print(f"Image size : {args.img_size}x{args.img_size}")
print(f"LR         : {args.lr}\n")

# ── Transforms ────────────────────────────────────────────────────────────────
rgb_train_transform = transforms.Compose([
    transforms.Resize((args.img_size + 16, args.img_size + 16)),
    transforms.RandomCrop(args.img_size),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

rgb_val_transform = transforms.Compose([
    transforms.Resize((args.img_size, args.img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

flow_transform = transforms.Compose([
    transforms.Resize((args.img_size, args.img_size)),
    transforms.ToTensor(),
    # flow normalized differently — zero mean
    transforms.Normalize([0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5]),
])

# ── Optical Flow computation ──────────────────────────────────────────────────
def compute_optical_flow(frames_bgr):
    """
    Compute optical flow from list of BGR numpy frames.
    Returns list of flow RGB images (for visualization and model input).
    Uses Farneback method — fast and good enough.
    """
    try:
        import cv2
        flow_frames = []
        prev_gray = None

        for frame in frames_bgr:
            if isinstance(frame, torch.Tensor):
                # convert tensor back to numpy for cv2
                f = (frame.permute(1,2,0).numpy() * 255).astype(np.uint8)
            else:
                f = frame

            gray = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) if len(f.shape)==3 else f

            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None,
                    pyr_scale=0.5, levels=3, winsize=15,
                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

                # flow magnitude and angle as 3-channel image
                mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                hsv = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)
                hsv[..., 0] = ang * 180 / np.pi / 2   # hue = direction
                hsv[..., 1] = 255                       # saturation
                hsv[..., 2] = cv2.normalize(mag, None, 0, 255,
                                            cv2.NORM_MINMAX)  # value = magnitude
                rgb_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
                flow_frames.append(Image.fromarray(rgb_flow))
            else:
                # first frame — zero flow
                flow_frames.append(Image.fromarray(
                    np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)))

            prev_gray = gray

        return flow_frames

    except ImportError:
        # cv2 not available — return zeros
        h, w = args.img_size, args.img_size
        return [Image.fromarray(np.zeros((h,w,3), dtype=np.uint8))
                for _ in frames_bgr]

# ── Dataset ───────────────────────────────────────────────────────────────────
class TwoStreamDataset(Dataset):
    """
    Returns two clips per video:
      rgb_clip:  [3, T, H, W] — RGB frames for appearance
      flow_clip: [3, T, H, W] — Optical flow for motion
    """
    def __init__(self, jpg_root, annotation_path, subset,
                 rgb_transform, n_frames=16):
        self.jpg_root      = jpg_root
        self.n_frames      = n_frames
        self.rgb_transform = rgb_transform
        self.is_train      = (subset == 'training')

        with open(annotation_path, 'r') as f:
            ann = json.load(f)

        self.samples = []
        for vid, info in ann['database'].items():
            if info['subset'] != subset: continue
            lbl = info['annotations']['label']
            idx = C2I.get(lbl, -1)
            if idx < 0: continue
            for cand in [
                os.path.join(jpg_root, lbl, vid),
                os.path.join(jpg_root, vid),
            ]:
                if os.path.isdir(cand):
                    self.samples.append((cand, idx))
                    break

        print(f"[{subset:10s}] {len(self.samples)} videos")

    def _load_frames_pil(self, vid_dir):
        """Load frames as PIL images."""
        files = sorted([f for f in os.listdir(vid_dir)
                        if f.endswith('.jpg')])
        total = len(files)
        if total == 0:
            return []
        indices = np.linspace(0, total-1, self.n_frames, dtype=int)
        frames  = []
        for i in indices:
            try:
                img = Image.open(os.path.join(vid_dir, files[i])).convert('RGB')
                frames.append(img)
            except:
                frames.append(Image.new('RGB', (args.img_size, args.img_size)))
        return frames

    def _frames_to_tensor(self, pil_frames, transform):
        """Convert PIL frame list → [3, T, H, W] tensor."""
        if not pil_frames:
            return torch.zeros(3, self.n_frames, args.img_size, args.img_size)
        tensors = [transform(f) for f in pil_frames]
        return torch.stack(tensors, 0).permute(1, 0, 2, 3)

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        vid_dir, label = self.samples[idx]

        # Load RGB frames
        pil_frames = self._load_frames_pil(vid_dir)
        rgb_clip   = self._frames_to_tensor(pil_frames, self.rgb_transform)

        # Compute optical flow from same frames
        flow_frames = compute_optical_flow(
            [np.array(f) for f in pil_frames])
        flow_clip   = self._frames_to_tensor(flow_frames, flow_transform)

        return [rgb_clip, flow_clip], label

def collate_fn(batch):
    rgb    = torch.stack([b[0][0] for b in batch])
    flow   = torch.stack([b[0][1] for b in batch])
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return [rgb, flow], labels

# ── Two-Stream Model ──────────────────────────────────────────────────────────
class TwoStreamModel(nn.Module):
    """
    Two separate R3D-18 encoders:
      rgb_encoder:  processes RGB frames (what is happening)
      flow_encoder: processes optical flow (how fast/energetic)

    Features are concatenated and classified together.
    This is the classic two-stream architecture from Simonyan et al.
    """
    def __init__(self, n_classes=5, fusion='concat'):
        super().__init__()
        self.fusion = fusion

        # RGB stream — pretrained on Kinetics
        rgb_base = r3d_18(weights=R3D_18_Weights.DEFAULT)
        self.rgb_encoder = nn.Sequential(*list(rgb_base.children())[:-1])

        # Flow stream — pretrained on Kinetics
        flow_base = r3d_18(weights=R3D_18_Weights.DEFAULT)
        self.flow_encoder = nn.Sequential(*list(flow_base.children())[:-1])

        # Fusion + classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.6),
            nn.Linear(512 + 512, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes)
        )

        print(f"✅ Two-Stream R3D-18 built")
        print(f"   RGB encoder   : R3D-18 (Kinetics pretrained)")
        print(f"   Flow encoder  : R3D-18 (Kinetics pretrained)")
        print(f"   Fusion        : {fusion} → 1024 → 512 → {n_classes}")

    def forward(self, inputs):
        rgb_clip, flow_clip = inputs

        # encode each stream
        rgb_feat  = self.rgb_encoder(rgb_clip)   # [B, 512, 1, 1, 1]
        flow_feat = self.flow_encoder(flow_clip)  # [B, 512, 1, 1, 1]

        # flatten
        rgb_feat  = rgb_feat.squeeze(-1).squeeze(-1).squeeze(-1)
        flow_feat = flow_feat.squeeze(-1).squeeze(-1).squeeze(-1)

        # concatenate
        fused = torch.cat([rgb_feat, flow_feat], dim=1)  # [B, 1024]
        return self.classifier(fused)

# ── Logging ───────────────────────────────────────────────────────────────────
def save_log(path, epoch, loss, acc, lr):
    header = not os.path.exists(path)
    with open(path, 'a') as f:
        if header: f.write("epoch\tloss\tacc\tlr\n")
        f.write(f"{epoch}\t{loss:.6f}\t{acc:.6f}\t{lr:.8f}\n")

def save_batch_log(path, epoch, batch, iteration, loss, acc, lr):
    header = not os.path.exists(path)
    with open(path, 'a') as f:
        if header: f.write("epoch\tbatch\titer\tloss\tacc\tlr\n")
        f.write(f"{epoch}\t{batch}\t{iteration}\t"
                f"{loss:.6f}\t{acc:.6f}\t{lr:.8f}\n")

# ── Train / Val ───────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, epoch, batch_log, lr):
    model.train()
    total_loss, correct, total = 0, 0, 0
    offset = (epoch - 1) * len(loader)

    for i, (inputs, labels) in enumerate(loader):
        inputs = [x.to(device) for x in inputs]
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
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
        for inputs, labels in loader:
            inputs = [x.to(device) for x in inputs]
            labels = labels.to(device)

            outputs = model(inputs)
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
        pct = 100 * class_correct[i] / max(1, class_total[i])
        per_class[CLASSES[i]] = f"{class_correct[i]}/{class_total[i]} ({pct:.1f}%)"

    return total_loss / len(loader), correct / total, per_class

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Loading datasets...")
    train_ds = TwoStreamDataset(args.jpg_root, args.annotation,
                                'training', rgb_train_transform, args.n_frames)
    val_ds   = TwoStreamDataset(args.jpg_root, args.annotation,
                                'validation', rgb_val_transform, args.n_frames)

    if len(train_ds) == 0:
        print("❌ No training samples!"); exit(1)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.n_workers,
                              collate_fn=collate_fn, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size,
                              shuffle=False, num_workers=args.n_workers,
                              collate_fn=collate_fn, pin_memory=True)

    print("\nBuilding Two-Stream model...")
    model = TwoStreamModel(n_classes=N_CLASSES).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {total_params/1e6:.2f}M\n")

    # class weights
    weights   = torch.tensor([1.61, 1.50, 1.03, 1.53, 1.17]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    # use lower lr for pretrained backbone, higher for classifier
    optimizer = torch.optim.SGD([
        {'params': model.rgb_encoder.parameters(),  'lr': args.lr * 0.1},
        {'params': model.flow_encoder.parameters(), 'lr': args.lr * 0.1},
        {'params': model.classifier.parameters(),   'lr': args.lr},
    ], momentum=0.9, weight_decay=1e-3, nesterov=True)

    scheduler = MultiStepLR(optimizer, milestones=[15, 30, 40], gamma=0.1)

    train_log = os.path.join(args.result_path, 'train.txt')
    val_log   = os.path.join(args.result_path, 'val.txt')
    batch_log = os.path.join(args.result_path, 'train_batch.txt')

    start_epoch  = 1
    best_val_acc = 0.0
    if args.resume_path and os.path.exists(args.resume_path):
        ckpt = torch.load(args.resume_path, map_location=device)
        model.load_state_dict(ckpt['state_dict'])
        if 'optimizer' in ckpt: optimizer.load_state_dict(ckpt['optimizer'])
        if 'scheduler' in ckpt: scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch  = ckpt.get('epoch', 1) + 1
        best_val_acc = ckpt.get('best_val_acc', 0.0)
        print(f"✅ Resumed from epoch {start_epoch-1}")

    import json as _json
    with open(os.path.join(args.result_path, 'opts.json'), 'w') as f:
        _json.dump(vars(args), f, indent=2)

    print(f"\n{'='*60}")
    print(f" Model      : Two-Stream R3D-18")
    print(f" Train vids : {len(train_ds)}")
    print(f" Val vids   : {len(val_ds)}")
    print(f" Epochs     : {args.n_epochs}")
    print(f" Batch size : {args.batch_size}")
    print(f" LR         : {args.lr} (backbone: {args.lr*0.1})")
    print(f" Classes    : {CLASSES}")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, args.n_epochs + 1):
        t0 = time.time()
        lr = optimizer.param_groups[-1]['lr']

        tr_loss, tr_acc = train_epoch(
            model, train_loader, optimizer,
            criterion, epoch, batch_log, lr)

        vl_loss, vl_acc, per_cls = val_epoch(model, val_loader, criterion)
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

        if epoch % args.checkpoint == 0:
            p = os.path.join(args.result_path, f'save_{epoch}.pth')
            torch.save({
                'epoch': epoch, 'model': 'two_stream',
                'state_dict': model.state_dict(),
                'optimizer':  optimizer.state_dict(),
                'scheduler':  scheduler.state_dict(),
                'best_val_acc': best_val_acc,
            }, p)
            print(f"  💾 Checkpoint: save_{epoch}.pth")

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            p = os.path.join(args.result_path, 'best_model.pth')
            torch.save({
                'epoch': epoch, 'model': 'two_stream',
                'state_dict': model.state_dict(),
                'val_acc': vl_acc,
            }, p)
            print(f"  🏆 Best val acc: {vl_acc*100:.2f}% → best_model.pth\n")

    print(f"\n✅ Done! Best Val Acc: {best_val_acc*100:.2f}%")
    print(f"   Results: {args.result_path}")
