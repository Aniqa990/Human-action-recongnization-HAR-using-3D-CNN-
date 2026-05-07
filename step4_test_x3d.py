"""
step4_test_x3d.py
Tests X3D/MobileNet models on FYP child safety test dataset.

Usage:
    # Test X3D (auto-detects from checkpoint)
    python step4_test_x3d.py \
        --model_path "H:/My Drive/FYP_DATA_jpg_raw/results_x3d/best_model.pth" \
        --jpg_root "H:/My Drive/FYP_TEST_JPG"

    # Test MobileNet (specify model_type manually)
    python step4_test_x3d.py \
        --model_path "H:/My Drive/FYP_DATA_jpg_raw/results_mobilenet/best_model.pth" \
        --jpg_root "H:/My Drive/FYP_TEST_JPG" \
        --model_type mobilenet

    # Full options
    python step4_test_x3d.py \
        --model_path "path/to/best_model.pth" \
        --jpg_root "path/to/test/frames" \
        --annotation "path/to/dataset.json" \
        --model_type mobilenet \
        --batch_size 16 \
        --n_frames 16 \
        --img_size 160
"""

import os, json, argparse
import numpy as np
from PIL import Image
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

parser = argparse.ArgumentParser()
parser.add_argument('--model_path',  type=str,
                    default='H:/My Drive/FYP_DATA_jpg_raw/results_x3d/best_model.pth')
parser.add_argument('--jpg_root',    type=str,
                    default='H:/My Drive/FYP_TEST_JPG')
parser.add_argument('--annotation',  type=str,
                    default='H:/My Drive/FYP_TEST_JPG/dataset.json')
parser.add_argument('--batch_size',  type=int,   default=16)
parser.add_argument('--n_workers',   type=int,   default=2)
parser.add_argument('--n_frames',    type=int,   default=16)
parser.add_argument('--img_size',    type=int,   default=160)
parser.add_argument('--model_type',  type=str,   default=None,
                    choices=['x3d_hub', 'x3d_ptv', 'mobilenet', 'r3d18_fallback'],
                    help='Model type. Auto-detect from checkpoint if not specified.')
parser.add_argument('--output_json', type=str,   default='test_results.json')
args = parser.parse_args()

CLASSES   = ['fight', 'Normal', 'unsafeClimb', 'unsafeJump', 'unsafeThrow', 'fall']
C2I       = {c: i for i, c in enumerate(CLASSES)}
N_CLASSES = len(CLASSES)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"\n{'='*70}")
print(f"  X3D TEST SCRIPT")
print(f"{'='*70}")
print(f"Model path    : {args.model_path}")
print(f"Test root     : {args.jpg_root}")
print(f"Annotation    : {args.annotation}")
print(f"Device        : {device}")
print(f"Batch size    : {args.batch_size}")
print(f"N frames      : {args.n_frames}")
print(f"Image size    : {args.img_size}x{args.img_size}")
print(f"{'='*70}\n")

# ── Transforms ────────────────────────────────────────────────────────────────
test_transform = transforms.Compose([
    transforms.Resize((args.img_size, args.img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.45, 0.45, 0.45],
                         [0.225, 0.225, 0.225]),
])

# ── Dataset ───────────────────────────────────────────────────────────────────
class VideoDataset(Dataset):
    def __init__(self, jpg_root, annotation_path,
                 transform, n_frames=16):
        self.jpg_root  = jpg_root
        self.n_frames  = n_frames
        self.transform = transform
        self.samples   = []

        # Try to load annotation
        if os.path.exists(annotation_path):
            with open(annotation_path, 'r') as f:
                ann = json.load(f)
            
            for vid, info in ann['database'].items():
                lbl = info['annotations']['label']
                idx = C2I.get(lbl, -1)
                if idx < 0:
                    print(f"[WARN] Unknown label: {lbl}")
                    continue
                for cand in [
                    os.path.join(jpg_root, lbl, vid),
                    os.path.join(jpg_root, vid),
                ]:
                    if os.path.isdir(cand):
                        self.samples.append((cand, idx, vid, lbl))
                        break
        else:
            # If no annotation, auto-discover: jpg_root/{label}/{video_id}
            print(f"[INFO] No annotation found, auto-discovering from directory structure...")
            if os.path.isdir(jpg_root):
                for label in CLASSES:
                    label_dir = os.path.join(jpg_root, label)
                    if os.path.isdir(label_dir):
                        idx = C2I[label]
                        for vid_id in os.listdir(label_dir):
                            vid_path = os.path.join(label_dir, vid_id)
                            if os.path.isdir(vid_path):
                                self.samples.append((vid_path, idx, vid_id, label))

        print(f"[TEST] Loaded {len(self.samples)} test videos\n")
        if len(self.samples) == 0:
            print("[ERROR] No test samples found!")
            print(f"        Expected structure:")
            print(f"          {jpg_root}/{label}/{video_id}/*.jpg")
            print(f"        OR annotation JSON with 'database' field")

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
            except Exception as e:
                img = Image.new('RGB',
                                (args.img_size, args.img_size), (0,0,0))
            frames.append(self.transform(img))

        return torch.stack(frames, 0).permute(1, 0, 2, 3)  # [3, T, H, W]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        vid_dir, label, vid_id, label_name = self.samples[idx]
        return self._load_clip(vid_dir), label, vid_id, label_name

# ── Model ─────────────────────────────────────────────────────────────────────
def build_model_by_type(model_type):
    """Build model architecture based on model_type"""
    
    if 'x3d' in model_type.lower():
        # X3D model
        try:
            print(f"  Building X3D from torch.hub...")
            model = torch.hub.load(
                'facebookresearch/pytorchvideo',
                'x3d_s',
                pretrained=True
            )
            in_features = model.blocks[-1].proj.in_features
            model.blocks[-1].proj = nn.Linear(in_features, N_CLASSES)
            print("  ✅ X3D-S loaded from torch.hub")
            return model
        except Exception as e:
            print(f"  [INFO] torch.hub failed: {e}")
            try:
                print(f"  Building X3D via pytorchvideo...")
                from pytorchvideo.models.x3d import create_x3d
                model = create_x3d(
                    input_clip_length=args.n_frames,
                    input_crop_size=args.img_size,
                    model_num_class=N_CLASSES,
                )
                print("  ✅ X3D built via pytorchvideo")
                return model
            except Exception as e2:
                print(f"  [ERROR] X3D build failed: {e2}")
                return None
    
    elif 'mobilenet' in model_type.lower():
        # MobileNet model
        try:
            print(f"  Building MobileNetV3...")
            from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights
            model = mvit_v2_s(weights=MViT_V2_S_Weights.DEFAULT)
            model.head = nn.Linear(model.head.in_features, N_CLASSES)
            print("  ✅ MobileNetV3 loaded")
            return model
        except Exception as e:
            print(f"  [INFO] MobileNetV3 failed: {e}")
            try:
                print(f"  Trying alternative MobileNet...")
                from torchvision.models.video import r3d_18, R3D_18_Weights
                model = r3d_18(weights=R3D_18_Weights.DEFAULT)
                model.fc = nn.Linear(model.fc.in_features, N_CLASSES)
                print("  ✅ R3D-18 fallback loaded")
                return model
            except Exception as e2:
                print(f"  [ERROR] MobileNet build failed: {e2}")
                return None
    
    elif 'r3d' in model_type.lower() or 'r2d' in model_type.lower():
        # ResNet models
        try:
            print(f"  Building {model_type}...")
            from torchvision.models.video import r3d_18, R3D_18_Weights
            model = r3d_18(weights=R3D_18_Weights.DEFAULT)
            model.fc = nn.Linear(model.fc.in_features, N_CLASSES)
            print(f"  ✅ {model_type} loaded")
            return model
        except Exception as e:
            print(f"  [ERROR] {model_type} build failed: {e}")
            return None
    
    else:
        print(f"  [WARN] Unknown model type: {model_type}")
        return None

def build_and_load_model(checkpoint_path):
    """Load checkpoint and build appropriate model architecture"""
    
    # First: check if checkpoint exists and get model_type
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return None, None
    
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    ckpt_model_type = ckpt.get('model_type', None)
    
    # Use --model_type if provided, otherwise use checkpoint's model_type
    if args.model_type:
        model_type = args.model_type
        print(f"  Model type (from args) : {model_type}")
    elif ckpt_model_type:
        model_type = ckpt_model_type
        print(f"  Model type (from checkpoint) : {model_type}")
    else:
        print(f"  [ERROR] model_type not found in checkpoint or args")
        print(f"  Use: --model_type mobilenet  or  --model_type x3d_hub")
        return None, None
    
    print(f"  Checkpoint epoch      : {ckpt.get('epoch', '?')}")
    print(f"  Checkpoint val acc    : {ckpt.get('val_acc', '?')}")
    
    # Second: build model based on model_type
    print(f"\nBuilding {model_type} model...")
    model = build_model_by_type(model_type)
    
    if model is None:
        print("[ERROR] Failed to build model architecture")
        return None, None
    
    model = model.to(device)
    
    # Third: load checkpoint state
    print(f"\nLoading model weights...")
    try:
        model.load_state_dict(ckpt['state_dict'], strict=True)
        print("✅ Model state loaded (strict=True)")
    except RuntimeError as e:
        print(f"⚠️  Loading with strict=False (architecture mismatch)")
        model.load_state_dict(ckpt['state_dict'], strict=False)
    
    return model, model_type

# ── Testing ───────────────────────────────────────────────────────────────────
def test(model, loader):
    """Evaluate model on test set"""
    model.eval()
    
    all_preds = []
    all_labels = []
    all_video_ids = []
    all_label_names = []
    
    class_correct = [0] * N_CLASSES
    class_total   = [0] * N_CLASSES
    correct = 0
    total = 0
    
    print("Testing...")
    print("-" * 70)

    with torch.no_grad():
        for i, (clips, labels, video_ids, label_names) in enumerate(loader):
            clips  = clips.to(device)
            labels = labels.to(device)

            outputs = model(clips)
            preds   = outputs.argmax(dim=1)

            # Metrics
            correct    += (preds == labels).sum().item()
            total      += labels.size(0)

            # Per-class
            for lbl, pred in zip(labels.cpu(), preds.cpu()):
                class_total[lbl] += 1
                if lbl == pred:
                    class_correct[lbl] += 1

            # Collect for detailed report
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_video_ids.extend(video_ids)
            all_label_names.extend(label_names)

            if (i+1) % 5 == 0 or (i+1) == len(loader):
                print(f"  [{i+1}/{len(loader)}] processed", end='\r')
    
    print()
    overall_acc = correct / total if total > 0 else 0.0
    
    # Per-class metrics
    per_class = {}
    for i in range(N_CLASSES):
        acc_pct = 100 * class_correct[i] / max(1, class_total[i])
        per_class[CLASSES[i]] = {
            'correct': class_correct[i],
            'total': class_total[i],
            'accuracy': acc_pct
        }

    return {
        'overall_accuracy': overall_acc,
        'per_class': per_class,
        'all_preds': all_preds,
        'all_labels': all_labels,
        'all_video_ids': all_video_ids,
        'all_label_names': all_label_names,
    }

# ── Confusion Matrix ──────────────────────────────────────────────────────────
def print_confusion_matrix(preds, labels):
    """Print confusion matrix"""
    cm = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
    for pred, label in zip(preds, labels):
        cm[label, pred] += 1
    
    print("\nConfusion Matrix:")
    print("  " + " ".join(f"{CLASSES[i][:8]:>10}" for i in range(N_CLASSES)))
    for i in range(N_CLASSES):
        print(f"{CLASSES[i]:>10}", end="")
        for j in range(N_CLASSES):
            print(f"{cm[i, j]:>10}", end="")
        print()
    
    return cm

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # Load model
    print("Loading model...")
    model, model_type = build_and_load_model(args.model_path)
    
    if model is None:
        print("❌ Failed to load model!")
        exit(1)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {total_params/1e6:.2f}M\n")

    # Load dataset
    print("Loading test dataset...")
    test_ds = VideoDataset(
        args.jpg_root, args.annotation,
        test_transform, args.n_frames)

    if len(test_ds) == 0:
        print("❌ No test samples found!")
        print(f"   Checked: {args.jpg_root}")
        print(f"   Annotation: {args.annotation}")
        exit(1)

    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.n_workers, pin_memory=True,
        persistent_workers=args.n_workers > 0)

    # Test
    print("\n" + "="*70)
    print(" TESTING")
    print("="*70 + "\n")
    
    results = test(model, test_loader)

    # Print results
    print("\n" + "="*70)
    print(" TEST RESULTS")
    print("="*70)
    print(f"\nOverall Accuracy: {results['overall_accuracy']*100:.2f}%\n")
    
    print("Per-Class Accuracy:")
    print("-" * 70)
    for cls, metrics in results['per_class'].items():
        acc = metrics['accuracy']
        correct = metrics['correct']
        total = metrics['total']
        marker = "✅" if acc >= 70 else ("⚠️" if acc >= 50 else "❌")
        print(f"  {marker} {cls:15s}: {correct:3d}/{total:3d} ({acc:6.2f}%)")
    
    # Confusion matrix
    cm = print_confusion_matrix(results['all_preds'], results['all_labels'])
    
    # Per-video results
    print("\n" + "="*70)
    print(" PER-VIDEO RESULTS")
    print("="*70 + "\n")
    
    results_by_class = defaultdict(list)
    for pred, label, vid_id, label_name in zip(
        results['all_preds'], 
        results['all_labels'],
        results['all_video_ids'],
        results['all_label_names']):
        
        pred_name = CLASSES[pred]
        correct = pred == label
        results_by_class[label_name].append({
            'video_id': vid_id,
            'true_label': label_name,
            'pred_label': pred_name,
            'correct': correct
        })
    
    for label_name in sorted(results_by_class.keys()):
        videos = results_by_class[label_name]
        correct_count = sum(1 for v in videos if v['correct'])
        print(f"\n{label_name} ({correct_count}/{len(videos)}):")
        for v in videos:
            marker = "✅" if v['correct'] else "❌"
            print(f"  {marker} {v['video_id']:30s} "
                  f"→ {v['pred_label']:15s}")
    
    # Save JSON results
    output_data = {
        'model_path': args.model_path,
        'model_type': model_type,
        'test_root': args.jpg_root,
        'total_params_M': total_params / 1e6,
        'overall_accuracy': results['overall_accuracy'],
        'per_class': results['per_class'],
        'confusion_matrix': cm.tolist(),
        'class_names': CLASSES,
    }
    
    with open(args.output_json, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✅ Test complete!")
    print(f"   Results saved to: {args.output_json}")
    print(f"   Overall Acc: {results['overall_accuracy']*100:.2f}%")
    print("="*70 + "\n")
