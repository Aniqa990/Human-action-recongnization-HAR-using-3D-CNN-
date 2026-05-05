import os
import cv2
import numpy as np
from vidaug import augmentors as va

# --- List of individual augmentors ---
AUGMENTORS = [
    (va.HorizontalFlip(), 'HorizontalFlip'),
    (va.Brightness(1.5), 'Brightness'),
    (va.GaussianNoise(20), 'GaussianNoise'),
    (va.TemporalJitter(2), 'TemporalJitter'),
]

def grayscale_augment(frames):
    """Convert RGB frames to grayscale and back to 3-channel RGB."""
    return [cv2.cvtColor(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB) for f in frames]

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

import argparse

parser = argparse.ArgumentParser(description='Batch augment videos under an input directory')
parser.add_argument('--input_dir', type=str, default=r"G:\My Drive\DATASET",
                    help='Root input folder containing action class subfolders')
parser.add_argument('--output_dir', type=str, default=r"G:\My Drive\Augmented_FYP_Data",
                    help='Output root folder for augmented videos')
parser.add_argument('--action', type=str, default='',
                    help='(Optional) single action/class folder to process (e.g. unsafeClimb)')
args = parser.parse_args()

input_dir = args.input_dir
output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

for action in os.listdir(input_dir):
    # if an action filter was provided, skip others
    if args.action and action != args.action:
        continue
    action_path = os.path.join(input_dir, action)
    if not os.path.isdir(action_path):
        continue
    
    out_action_path = os.path.join(output_dir, action)
    os.makedirs(out_action_path, exist_ok=True)
    
    for video_file in os.listdir(action_path):
        # --- NEW CHECK STARTS HERE ---
        # Skip desktop.ini and any hidden files (starting with '.')
        if video_file.lower() == 'desktop.ini' or video_file.startswith('.'):
            continue
            
        video_path = os.path.join(action_path, video_file)
        
        # Skip if it's a sub-directory inside the action folder
        if os.path.isdir(video_path):
            continue
        # --- NEW CHECK ENDS HERE ---

        video_basename = os.path.splitext(video_file)[0]
        
        print(f"Checking {video_path} ...")
        clip = None
        fps = 25

        # --- Standard augmentors loop ---
        for aug, aug_name in AUGMENTORS:
            out_file = os.path.join(out_action_path, f"aug_{aug_name}_{video_basename}.mp4")
            
            if os.path.exists(out_file):
                print(f"  >> Skipping {aug_name}: File already exists.")
                continue

            if clip is None:
                clip, fps = load_video(video_path)
                if not clip or not isinstance(clip[0], (np.ndarray,)):
                    print(f"  !! Error: Could not load frames for {video_path}")
                    break

            try:
                aug_clip = aug(clip)
                aug_clip = [np.clip(f, 0, 255).astype(np.uint8) for f in aug_clip]
                save_video(aug_clip, out_file, fps=fps)
                print(f"  [+] Saved: {out_file}")
            except Exception as e:
                print(f"  [!] Error with {aug_name}: {e}")

        # --- Grayscale augmentor ---
        out_file_gray = os.path.join(out_action_path, f"aug_Grayscale_{video_basename}.mp4")
        
        if os.path.exists(out_file_gray):
            print(f"  >> Skipping Grayscale: File already exists.")
        else:
            if clip is None: 
                clip, fps = load_video(video_path)
            
            if clip:
                try:
                    gray_clip = grayscale_augment(clip)
                    save_video(gray_clip, out_file_gray, fps=fps)
                    print(f"  [+] Saved: {out_file_gray}")
                except Exception as e:
                    print(f"  [!] Error with Grayscale: {e}")