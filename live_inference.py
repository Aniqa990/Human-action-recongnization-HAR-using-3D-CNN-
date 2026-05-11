"""
live_inference.py
Real-time child safety action recognition using IP Webcam.
Supports MobileNet3D, X3D, and SlowFast — auto-detects from checkpoint.

Setup:
  1. Install IP Webcam app on Android phone
  2. Open app → Start Server
  3. Note IP address (e.g. http://192.168.1.5:8080)
  4. Make sure PC and phone are on same WiFi

Usage:
    # MobileNet3D
    python live_inference.py \
        --checkpoint /path/to/results_mobilenet/best_model.pth \
        --stream_url http://192.168.1.5:8080/video

    # X3D
    python live_inference.py \
        --checkpoint /path/to/results_x3d/best_model.pth \
        --stream_url http://192.168.1.5:8080/video

    # SlowFast
    python live_inference.py \
        --checkpoint /path/to/results_slowfast/best_model.pth \
        --stream_url http://192.168.1.5:8080/video

    # Test with webcam (no phone needed)
    python live_inference.py \
        --checkpoint /path/to/best_model.pth \
        --use_webcam

Requirements:
    pip install opencv-python torch torchvision pillow numpy requests
"""

import cv2
import sys
import time
import argparse
import threading
import collections
import numpy as np
from PIL import Image
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as transforms

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint',   type=str, required=True,
                    help='Path to best_model.pth (any model type)')
parser.add_argument('--stream_url',   type=str,
                    default='http://192.168.1.5:8080/video',
                    help='IP Webcam stream URL')
parser.add_argument('--use_webcam',   action='store_true',
                    help='Use local webcam instead of IP Webcam')
parser.add_argument('--webcam_id',    type=int, default=0,
                    help='Webcam device ID (default 0)')
parser.add_argument('--n_frames',     type=int, default=16,
                    help='Frames per clip for inference')
parser.add_argument('--img_size',     type=int, default=112,
                    help='Frame resize size')
parser.add_argument('--slow_frames',  type=int, default=8,
                    help='Slow pathway frames (SlowFast only)')
parser.add_argument('--fast_frames',  type=int, default=32,
                    help='Fast pathway frames (SlowFast only)')
parser.add_argument('--stride',       type=int, default=8,
                    help='Sliding window stride (frames to shift)')
parser.add_argument('--smooth',       type=int, default=3,
                    help='Smooth predictions over N clips')
parser.add_argument('--conf_thresh',  type=float, default=0.5,
                    help='Min confidence to show prediction')
parser.add_argument('--alert_count',  type=int, default=3,
                    help='Alert after N consecutive unsafe predictions')
args = parser.parse_args()

# ── Classes & Colors ──────────────────────────────────────────────────────────
CLASSES = ['fight', 'Normal', 'unsafeClimb', 'unsafeJump', 'unsafeThrow', 'fall']
N_CLS   = len(CLASSES)

SAFE_CLASSES   = {'Normal'}
UNSAFE_CLASSES = {'fight', 'unsafeClimb', 'unsafeJump', 'unsafeThrow', 'fall'}

# BGR colors for OpenCV
CLASS_COLORS = {
    'Normal'      : (0,   200,  0),    # Green
    'fight'       : (0,   0,   255),   # Red
    'unsafeClimb' : (0,   100, 255),   # Orange
    'unsafeJump'  : (0,   165, 255),   # Orange-red
    'unsafeThrow' : (255, 0,   150),   # Purple
    'fall'        : (0,   0,   200),   # Dark red
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice     : {device}")
print(f"Checkpoint : {args.checkpoint}\n")

# ── Transforms ────────────────────────────────────────────────────────────────
frame_transform = transforms.Compose([
    transforms.Resize((args.img_size, args.img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ── Model Definitions ─────────────────────────────────────────────────────────

# ── MobileNet3D ────────────────────────────────────────────────────────────
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
        self.dw = ConvBnRelu(in_ch, in_ch, (3,3,3),
                             (1,stride,stride), (1,1,1), in_ch)
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

# ── Two-Stream (SlowFast fallback) ────────────────────────────────────────
class TwoStreamModel(nn.Module):
    def __init__(self, n_classes=6):
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

# ── Load Model (auto-detect type) ─────────────────────────────────────────────
def load_model(ckpt_path):
    ckpt       = torch.load(ckpt_path, map_location=device)
    state      = ckpt.get('state_dict', ckpt)
    model_type = ckpt.get('model', ckpt.get('model_type', 'unknown'))

    print(f"Checkpoint model_type: '{model_type}'")

    model = None

    # ── Try MobileNet3D ──────────────────────────────────────────────────
    if 'mobilenet' in model_type.lower() or model_type == 'unknown':
        try:
            m = MobileNet3D(n_classes=N_CLS)
            ns = {k.replace('module.',''): v for k,v in state.items()}
            m.load_state_dict(ns, strict=True)
            model = m
            model_type = 'mobilenet3d'
            print("✅ Loaded as MobileNet3D")
        except Exception as e:
            print(f"   MobileNet3D failed: {e}")

    # ── Try X3D via torch.hub ────────────────────────────────────────────
    if model is None or 'x3d' in model_type.lower():
        for x3d_variant in ['x3d_s', 'x3d_xs', 'x3d_m']:
            try:
                m = torch.hub.load('facebookresearch/pytorchvideo',
                                   x3d_variant, pretrained=False)
                in_feat = m.blocks[-1].proj.in_features
                m.blocks[-1].proj = nn.Linear(in_feat, N_CLS)
                ns = {k.replace('module.',''): v for k,v in state.items()}
                m.load_state_dict(ns, strict=False)
                model = m
                model_type = f'x3d_{x3d_variant}'
                print(f"✅ Loaded as X3D ({x3d_variant})")
                break
            except Exception as e:
                print(f"   X3D {x3d_variant} failed: {e}")

    # ── Try SlowFast via torch.hub ───────────────────────────────────────
    if model is None or 'slowfast' in model_type.lower():
        try:
            m = torch.hub.load('facebookresearch/pytorchvideo:main',
                               'slowfast_r50', pretrained=False)
            in_feat = m.blocks[-1].proj.in_features
            m.blocks[-1].proj = nn.Linear(in_feat, N_CLS)
            ns = {k.replace('module.',''): v for k,v in state.items()}
            m.load_state_dict(ns, strict=False)
            model = m
            model_type = 'slowfast'
            print("✅ Loaded as SlowFast-R50")
        except Exception as e:
            print(f"   SlowFast failed: {e}")

    # ── Try Two-Stream fallback ──────────────────────────────────────────
    if model is None:
        try:
            m = TwoStreamModel(n_classes=N_CLS)
            ns = {k.replace('module.',''): v for k,v in state.items()}
            m.load_state_dict(ns, strict=False)
            model = m
            model_type = 'two_stream'
            print("✅ Loaded as Two-Stream (SlowFast fallback)")
        except Exception as e:
            print(f"   Two-Stream failed: {e}")

    if model is None:
        raise RuntimeError("❌ Could not load model with any architecture!")

    model = model.to(device)
    model.eval()

    ep  = ckpt.get('epoch', '?')
    acc = ckpt.get('val_acc', ckpt.get('best_val_acc', 0))
    print(f"   Epoch: {ep}  |  Val Acc: {acc*100:.2f}%\n")

    return model, model_type

# ── Clip preparation ───────────────────────────────────────────────────────────
def frames_to_tensor(frames, n):
    """Convert list of BGR frames → [3, n, H, W] tensor."""
    total   = len(frames)
    indices = np.linspace(0, total-1, n, dtype=int)
    clips   = []
    for i in indices:
        img = Image.fromarray(cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB))
        clips.append(frame_transform(img))
    return torch.stack(clips, 0).permute(1, 0, 2, 3).unsqueeze(0)  # [1,3,T,H,W]

def prepare_input(frames, model_type):
    """Prepare input based on model type."""
    if 'slowfast' in model_type or 'two_stream' in model_type:
        slow = frames_to_tensor(frames, args.slow_frames).to(device)
        fast = frames_to_tensor(frames, args.fast_frames).to(device)
        return [slow, fast]
    else:
        return frames_to_tensor(frames, args.n_frames).to(device)

# ── Inference ──────────────────────────────────────────────────────────────────
def run_inference(model, model_type, frames):
    with torch.no_grad():
        inp     = prepare_input(frames, model_type)
        outputs = model(inp)
        probs   = torch.softmax(outputs, dim=1)[0].cpu().numpy()
    pred_idx = probs.argmax()
    return CLASSES[pred_idx], probs[pred_idx], probs

# ── Display helpers ────────────────────────────────────────────────────────────
def draw_overlay(frame, pred_class, confidence, all_probs,
                 fps, alert_active, model_type, smoothed_preds):
    h, w = frame.shape[:2]

    # semi-transparent dark bar at top
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 120), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # prediction color
    color = CLASS_COLORS.get(pred_class, (255, 255, 255))
    is_unsafe = pred_class in UNSAFE_CLASSES

    # main prediction text
    cv2.putText(frame, f"Action: {pred_class}",
                (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                color, 2, cv2.LINE_AA)

    # confidence
    conf_color = (0, 200, 0) if confidence >= 0.7 else \
                 (0, 165, 255) if confidence >= 0.5 else (0, 0, 255)
    cv2.putText(frame, f"Confidence: {confidence*100:.1f}%",
                (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                conf_color, 2, cv2.LINE_AA)

    # model type + FPS
    cv2.putText(frame, f"Model: {model_type}  |  FPS: {fps:.1f}",
                (15, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (180, 180, 180), 1, cv2.LINE_AA)

    # ALERT banner
    if alert_active and is_unsafe:
        alert_overlay = frame.copy()
        cv2.rectangle(alert_overlay, (0, h-70), (w, h), (0, 0, 200), -1)
        cv2.addWeighted(alert_overlay, 0.7, frame, 0.3, 0, frame)
        cv2.putText(frame, f"⚠ UNSAFE ACTION DETECTED: {pred_class.upper()} ⚠",
                    (15, h-25), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                    (255, 255, 255), 2, cv2.LINE_AA)

    # probability bars (right side)
    bar_x = w - 220
    bar_y = 10
    for i, (cls, prob) in enumerate(zip(CLASSES, all_probs)):
        c = CLASS_COLORS.get(cls, (200, 200, 200))
        bar_len = int(prob * 150)
        cv2.rectangle(frame, (bar_x, bar_y + i*22),
                      (bar_x + bar_len, bar_y + i*22 + 16), c, -1)
        cv2.putText(frame, f"{cls[:11]:11s} {prob*100:4.1f}%",
                    (bar_x - 145, bar_y + i*22 + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (220, 220, 220), 1, cv2.LINE_AA)

    # smoothed prediction history (bottom left)
    if smoothed_preds:
        cv2.putText(frame, "Recent:",
                    (15, h - 80 if alert_active else h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (180, 180, 180), 1, cv2.LINE_AA)
        history_str = " → ".join(list(smoothed_preds)[-5:])
        cv2.putText(frame, history_str,
                    (75, h - 80 if alert_active else h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (200, 200, 200), 1, cv2.LINE_AA)

    # border color — red if unsafe
    if is_unsafe and confidence >= args.conf_thresh:
        cv2.rectangle(frame, (0, 0), (w-1, h-1), color, 4)

    return frame

# ── Stream reader (runs in background thread) ─────────────────────────────────
class StreamReader:
    def __init__(self, source):
        self.source  = source
        self.cap     = None
        self.frame   = None
        self.running = False
        self.lock    = threading.Lock()

    def start(self):
        if isinstance(self.source, int):
            self.cap = cv2.VideoCapture(self.source)
        else:
            self.cap = cv2.VideoCapture(self.source)

        if not self.cap.isOpened():
            raise RuntimeError(f"❌ Cannot open stream: {self.source}")

        self.running = True
        t = threading.Thread(target=self._read_loop, daemon=True)
        t.start()
        print(f"✅ Stream opened: {self.source}")

    def _read_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame
            else:
                time.sleep(0.01)

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()

# ── Main live inference loop ───────────────────────────────────────────────────
def main():
    # Load model
    model, model_type = load_model(args.checkpoint)

    # Determine buffer size based on model
    if 'slowfast' in model_type or 'two_stream' in model_type:
        buffer_size = max(args.slow_frames, args.fast_frames)
    else:
        buffer_size = args.n_frames

    print(f"Model type  : {model_type}")
    print(f"Buffer size : {buffer_size} frames")
    print(f"Stride      : {args.stride} frames")
    print(f"Smooth over : {args.smooth} predictions")
    print(f"Alert after : {args.alert_count} consecutive unsafe\n")

    # Open stream
    source = args.webcam_id if args.use_webcam else args.stream_url
    reader = StreamReader(source)
    try:
        reader.start()
    except RuntimeError as e:
        print(e)
        print("\nTroubleshooting:")
        print("  1. Is IP Webcam app running on phone?")
        print("  2. Are phone and PC on same WiFi?")
        print("  3. Try --use_webcam to test with local camera")
        sys.exit(1)

    # State
    frame_buffer     = []
    pred_history     = collections.deque(maxlen=args.smooth)
    pred_class_hist  = collections.deque(maxlen=args.alert_count)
    current_pred     = "Initializing..."
    current_conf     = 0.0
    current_probs    = np.ones(N_CLS) / N_CLS
    alert_active     = False
    frames_since_inf = 0
    fps_timer        = time.time()
    fps              = 0.0
    frame_count      = 0

    print("🎥 Live inference running...")
    print("   Press Q to quit")
    print("   Press S to save current frame\n")

    while True:
        frame = reader.read()
        if frame is None:
            time.sleep(0.01)
            continue

        frame_count  += 1
        frames_since_inf += 1

        # Add to buffer
        frame_buffer.append(frame.copy())
        if len(frame_buffer) > buffer_size:
            frame_buffer.pop(0)

        # Run inference when buffer is full and stride reached
        if (len(frame_buffer) >= buffer_size and
                frames_since_inf >= args.stride):
            frames_since_inf = 0

            try:
                pred, conf, probs = run_inference(
                    model, model_type, frame_buffer)

                # Update prediction history
                pred_history.append(pred)
                pred_class_hist.append(pred)

                # Smooth: most common prediction in recent history
                if len(pred_history) > 0:
                    from collections import Counter
                    smoothed = Counter(pred_history).most_common(1)[0][0]
                    smoothed_conf = np.mean(
                        [probs[CLASSES.index(p)] for p in pred_history
                         if p == smoothed])
                    current_pred  = smoothed
                    current_conf  = smoothed_conf
                    current_probs = probs

                # Alert logic
                if len(pred_class_hist) >= args.alert_count:
                    recent = list(pred_class_hist)[-args.alert_count:]
                    if (len(set(recent)) == 1 and
                            recent[0] in UNSAFE_CLASSES):
                        if not alert_active:
                            print(f"\n🚨 ALERT: {recent[0]} detected!")
                        alert_active = True
                    else:
                        alert_active = False

            except Exception as e:
                print(f"Inference error: {e}")

        # FPS calculation
        if frame_count % 30 == 0:
            fps = 30 / (time.time() - fps_timer)
            fps_timer = time.time()

        # Draw overlay
        display = draw_overlay(
            frame.copy(),
            current_pred,
            current_conf,
            current_probs,
            fps,
            alert_active,
            model_type,
            list(pred_history)
        )

        # Show
        cv2.imshow('FYP - Child Safety Monitor', display)

        # Key handling
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\nQuitting...")
            break
        elif key == ord('s'):
            fname = f"capture_{int(time.time())}.jpg"
            cv2.imwrite(fname, display)
            print(f"Saved: {fname}")

    reader.stop()
    cv2.destroyAllWindows()
    print("Done.")

if __name__ == '__main__':
    main()
