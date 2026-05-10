import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import argparse
from PIL import Image
import torchvision.transforms as transforms

# ── CONFIG & MODEL BUILDER ──────────────────────────────────────────────────
def build_slowfast_model(checkpoint_path, n_classes=5):
    """Rebuilds the SlowFast architecture and loads weights."""
    import torch.hub
    # Load the base model structure
    model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=False)
    
    # Update the head to match your training (5 classes)
    in_features = model.blocks[-1].proj.in_features
    model.blocks[-1].proj = nn.Linear(in_features, n_classes)
    
    # Load your specific weights
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model

# ── VIDEO PROCESSING ────────────────────────────────────────────────────────
def process_video(video_path, slow_frames=8, fast_frames=32, img_size=224):
    """Extracts frames from a video and prepares the Slow/Fast tensors."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR (OpenCV) to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame))
    cap.release()

    if len(frames) == 0:
        raise ValueError("Could not read any frames from the video.")

    # Normalization used in your training
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.45, 0.45, 0.45], [0.225, 0.225, 0.225]),
    ])

    # Sample frames for Slow and Fast pathways
    def sample_frames(frame_list, count):
        indices = np.linspace(0, len(frame_list) - 1, count, dtype=int)
        sampled = [transform(frame_list[i]) for i in indices]
        return torch.stack(sampled, 0).permute(1, 0, 2, 3) # [C, T, H, W]

    slow_clip = sample_frames(frames, slow_frames)
    fast_clip = sample_frames(frames, fast_frames)

    return [slow_clip.unsqueeze(0), fast_clip.unsqueeze(0)] # Add batch dim

# ── MAIN EXECUTION ──────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="SlowFast Single Video Inference")
    parser.add_argument("--video", type=str, required=True, help="Path to the unseen .mp4 video")
    parser.add_argument("--model", type=str, required=True, help="Path to best_model.pth")
    args = parser.parse_args()

    CLASSES = ['fight', 'unsafeClimb', 'unsafeJump', 'unsafeThrow', 'fall']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"📦 Loading Model: {os.path.basename(args.model)}")
    model = build_slowfast_model(args.model).to(device)

    print(f"🎬 Processing Video: {os.path.basename(args.video)}")
    try:
        inputs = process_video(args.video)
        inputs = [x.to(device) for x in inputs]

        with torch.no_grad():
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)[0]
            pred_idx = torch.argmax(probs).item()
            confidence = probs[pred_idx].item()

        print("\n" + "="*30)
        print(f"RESULT: {CLASSES[pred_idx].upper()}")
        print(f"CONFIDENCE: {confidence*100:.2f}%")
        print("="*30)

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
