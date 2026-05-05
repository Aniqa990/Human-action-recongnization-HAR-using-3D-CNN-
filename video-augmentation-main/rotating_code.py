import os
import cv2
import numpy as np
from vidaug import augmentors as va

# --- Augmenter Settings ---
AUGMENTOR = (va.RandomRotate(degrees=45), 'RandomRotate')

# Using raw strings to handle Windows paths and spaces in "My Drive"
input_dir = r"G:\My Drive\DATASET"
output_dir = r"G:\My Drive\Augmented_FYP_Data"

os.makedirs(output_dir, exist_ok=True)

def load_video(video_path):
    """Load video and return list of frames (RGB)."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return frames, fps if fps > 0 else 25

def save_video(frames, out_path, fps=25):
    """Save list of frames (RGB) as video."""
    if not frames:
        return
    h, w, _ = frames[0].shape
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for frame in frames:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    out.release()

for action in os.listdir(input_dir):
    action_path = os.path.join(input_dir, action)
    if not os.path.isdir(action_path):
        continue
    
    out_action_path = os.path.join(output_dir, action)
    os.makedirs(out_action_path, exist_ok=True)
    
    for video_file in os.listdir(action_path):
        # --- NEW CHECK: Skip desktop.ini and hidden files ---
        if video_file.lower() == 'desktop.ini' or video_file.startswith('.'):
            continue

        video_path = os.path.join(action_path, video_file)
        
        # --- NEW CHECK: Skip if it's a folder instead of a file ---
        if os.path.isdir(video_path):
            continue

        base_name = os.path.splitext(video_file)[0]
        aug, aug_name = AUGMENTOR
        
        # Construct the output filename
        out_file = os.path.join(out_action_path, f"aug_{aug_name}_{base_name}.mp4")

        # ✅ CHECK IF FILE ALREADY EXISTS
        if os.path.exists(out_file):
            print(f"Skipping {video_file}: {aug_name} already exists.")
            continue

        print(f"Processing {video_path} ...")
        
        # Load video ONLY if the output doesn't exist
        clip, fps = load_video(video_path)
        if not clip:
            # This print will no longer trigger for desktop.ini thanks to the check above
            print(f"Skipping {video_path}: could not load frames.")
            continue

        try:
            aug_clip = aug(clip)
            processed_frames = []
            for f in aug_clip:
                if f.dtype != np.uint8:
                    f = (f * 255).clip(0, 255).astype(np.uint8)
                processed_frames.append(f)
            
            save_video(processed_frames, out_file, fps=fps)
            print(f"Saved: {out_file}")
        except Exception as e:
            print(f"Error augmenting {video_path}: {e}")