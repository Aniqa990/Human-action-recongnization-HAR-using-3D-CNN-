"""
STEP 1: generate_video_jpgs.py
Converts raw .mp4 videos → JPG frames folder structure.
NO segmentation — keeps background for height/context cues.

Output structure:
    jpg_root/
        fight/
            video1/
                img_00001.jpg
                img_00002.jpg
        Normal/
            video1/
                ...

Usage:
    python generate_video_jpgs.py \
        --video_root "G:/My Drive/Segmented_FYP_DATA" \
        --jpg_root   "G:/My Drive/Segmented_FYP_DATA_jpg_raw"
"""

import os
import subprocess
import argparse
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--video_root', type=str,
                    default='G:/My Drive/Segmented_FYP_DATA',
                    help='Root folder with class subfolders containing videos')
parser.add_argument('--jpg_root', type=str,
                    default='G:/My Drive/Segmented_FYP_DATA_jpg_raw',
                    help='Output root for JPG frames')
parser.add_argument('--fps', type=int, default=10,
                    help='Frames per second to extract (10 is good balance)')
parser.add_argument('--img_size', type=int, default=128,
                    help='Resize shorter side to this (128 or 112)')
parser.add_argument('--n_workers', type=int, default=4)
parser.add_argument('--ffmpeg_path', type=str, default='',
                    help='Optional absolute path to ffmpeg executable')
args = parser.parse_args()

CLASSES = ['fight', 'Normal', 'unsafeClimb', 'unsafeJump', 'unsafeThrow', 'fall']
VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv'}


def resolve_ffmpeg(ffmpeg_path: str = ''):
    if ffmpeg_path and Path(ffmpeg_path).exists():
        return ffmpeg_path

    from_path = shutil.which('ffmpeg')
    if from_path:
        return from_path

    winget_links = Path.home() / 'AppData' / 'Local' / 'Microsoft' / 'WinGet' / 'Links' / 'ffmpeg.exe'
    if winget_links.exists():
        return str(winget_links)

    pkg_root = Path.home() / 'AppData' / 'Local' / 'Microsoft' / 'WinGet' / 'Packages'
    if pkg_root.exists():
        ffmpeg_bins = sorted(pkg_root.glob('Gyan.FFmpeg_*/*/bin/ffmpeg.exe'))
        if ffmpeg_bins:
            return str(ffmpeg_bins[-1])

    return ''


FFMPEG_PATH = resolve_ffmpeg(args.ffmpeg_path)


def summarize_stderr(stderr, max_len=180):
    if not stderr:
        return 'no stderr from ffmpeg'
    lines = [line.strip() for line in stderr.splitlines() if line.strip()]
    if not lines:
        return 'no stderr from ffmpeg'
    summary = lines[-1]
    if len(summary) > max_len:
        summary = summary[:max_len] + '...'
    return summary

def is_video(path):
    return Path(path).suffix.lower() in VIDEO_EXTS

def convert_video(job):
    src_video, dst_dir = job
    if os.path.exists(dst_dir):
        existing = [f for f in os.listdir(dst_dir) if f.endswith('.jpg')]
        if len(existing) > 0:
            return f"SKIP (already done): {dst_dir}"

    os.makedirs(dst_dir, exist_ok=True)

    src_video = src_video.replace('\\', '/')

    if not os.path.isfile(src_video):
        return f"MISSING_SOURCE: {src_video}"

    if not FFMPEG_PATH:
        return f"MISSING_FFMPEG: ffmpeg not found in PATH ({src_video})"

    cmd = [
        FFMPEG_PATH, '-y',
        '-i', src_video,
        '-vf', f'scale=-1:{args.img_size}',
        '-r', str(args.fps),
        '-q:v', '2',
        os.path.join(dst_dir, 'img_%05d.jpg')
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            return f"FFMPEG_FAIL (code={result.returncode}) {summarize_stderr(result.stderr)}: {src_video}"
        n = len([f for f in os.listdir(dst_dir) if f.endswith('.jpg')])
        if n == 0:
            return f"FAIL (0 frames): {src_video}"
        return f"OK ({n} frames): {Path(src_video).name}"
    except subprocess.TimeoutExpired:
        return f"TIMEOUT: {src_video}"
    except FileNotFoundError:
        return f"MISSING_FFMPEG: ffmpeg executable not found ({src_video})"
    except Exception as e:
        return f"ERROR {e}: {src_video}"

# collect jobs
jobs = []
skipped_classes = []
for cls in CLASSES:
    cls_dir = os.path.join(args.video_root, cls)
    if not os.path.exists(cls_dir):
        skipped_classes.append(cls)
        continue
    for fname in os.listdir(cls_dir):
        fpath = os.path.join(cls_dir, fname).replace('\\', '/')
        if is_video(fpath):
            video_name = Path(fname).stem
            dst = os.path.join(args.jpg_root, cls, video_name).replace('\\', '/')
            jobs.append((fpath, dst))

print(f"Total videos: {len(jobs)}")
if skipped_classes:
    print(f"[WARN] Class folders not found: {skipped_classes}")
if not FFMPEG_PATH:
    print("[WARN] ffmpeg is not available in PATH. Install ffmpeg or add it to PATH.")

failed = []
with ThreadPoolExecutor(max_workers=args.n_workers) as ex:
    for i, result in enumerate(ex.map(convert_video, jobs)):
        status = "✅" if result.startswith("OK") else ("⏭️" if result.startswith("SKIP") else "❌")
        print(f"[{i+1}/{len(jobs)}] {status} {result}")
        if not (result.startswith("OK") or result.startswith("SKIP")):
            failed.append(result)

print(f"\n✅ Done. Frames saved to: {args.jpg_root}")
print(f"   Failed: {len(failed)}")
for f in failed:
    print(f"   {f}")
