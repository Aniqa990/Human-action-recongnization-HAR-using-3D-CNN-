# import subprocess
# import argparse
# from pathlib import Path

# from joblib import Parallel, delayed


# def video_process(video_file_path, dst_root_path, ext, fps=-1, size=240):
#     if ext != video_file_path.suffix:
#         return

#     ffprobe_cmd = ('ffprobe -v error -select_streams v:0 '
#                    '-of default=noprint_wrappers=1:nokey=1 -show_entries '
#                    'stream=width,height,avg_frame_rate,duration').split()
#     ffprobe_cmd.append(str(video_file_path))

#     p = subprocess.run(ffprobe_cmd, capture_output=True)
#     res = p.stdout.decode('utf-8').splitlines()
#     if len(res) < 4:
#         return

#     frame_rate = [float(r) for r in res[2].split('/')]
#     frame_rate = frame_rate[0] / frame_rate[1]
#     duration = float(res[3])
#     n_frames = int(frame_rate * duration)

#     name = video_file_path.stem
#     dst_dir_path = dst_root_path / name
#     dst_dir_path.mkdir(exist_ok=True)
#     n_exist_frames = len([
#         x for x in dst_dir_path.iterdir()
#         if x.suffix == '.jpg' and x.name[0] != '.'
#     ])

#     if n_exist_frames >= n_frames:
#         return

#     width = int(res[0])
#     height = int(res[1])

#     if width > height:
#         vf_param = 'scale=-1:{}'.format(size)
#     else:
#         vf_param = 'scale={}:-1'.format(size)

#     if fps > 0:
#         vf_param += ',minterpolate={}'.format(fps)

#     ffmpeg_cmd = ['ffmpeg', '-i', str(video_file_path), '-vf', vf_param]
#     ffmpeg_cmd += ['-threads', '1', '{}/image_%05d.jpg'.format(dst_dir_path)]
#     print(ffmpeg_cmd)
#     subprocess.run(ffmpeg_cmd)
#     print('\n')


# def class_process(class_dir_path, dst_root_path, ext, fps=-1, size=240):
#     if not class_dir_path.is_dir():
#         return

#     dst_class_path = dst_root_path / class_dir_path.name
#     dst_class_path.mkdir(exist_ok=True)

#     for video_file_path in sorted(class_dir_path.iterdir()):
#         video_process(video_file_path, dst_class_path, ext, fps, size)


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         'dir_path', default=None, type=Path, help='Directory path of videos')
#     parser.add_argument(
#         'dst_path',
#         default=None,
#         type=Path,
#         help='Directory path of jpg videos')
#     parser.add_argument(
#         'dataset',
#         default='',
#         type=str,
#         help='Dataset name (kinetics | mit | ucf101 | hmdb51 | activitynet)')
#     parser.add_argument(
#         '--n_jobs', default=-1, type=int, help='Number of parallel jobs')
#     parser.add_argument(
#         '--fps',
#         default=-1,
#         type=int,
#         help=('Frame rates of output videos. '
#               '-1 means original frame rates.'))
#     parser.add_argument(
#         '--size', default=240, type=int, help='Frame size of output videos.')
#     args = parser.parse_args()

#     if args.dataset in ['kinetics', 'mit', 'activitynet']:
#         ext = '.mp4'
#     else:
#         ext = '.avi'

#     if args.dataset == 'activitynet':
#         video_file_paths = [x for x in sorted(args.dir_path.iterdir())]
#         status_list = Parallel(
#             n_jobs=args.n_jobs,
#             backend='threading')(delayed(video_process)(
#                 video_file_path, args.dst_path, ext, args.fps, args.size)
#                                  for video_file_path in video_file_paths)
#     else:
#         class_dir_paths = [x for x in sorted(args.dir_path.iterdir())]
#         test_set_video_path = args.dir_path / 'test'
#         if test_set_video_path.exists():
#             class_dir_paths.append(test_set_video_path)

#         status_list = Parallel(
#             n_jobs=args.n_jobs,
#             backend='threading')(delayed(class_process)(
#                 class_dir_path, args.dst_path, ext, args.fps, args.size)
#                                  for class_dir_path in class_dir_paths)

import subprocess
import argparse
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


# MP4/video magic byte signatures to detect extension-less video files
VIDEO_MAGIC_BYTES = [
    b'\x00\x00\x00\x18ftyp',  # mp4
    b'\x00\x00\x00\x20ftyp',  # mp4
    b'\x00\x00\x00\x1cftyp',  # mp4
    b'ftyp',                   # mp4 (offset 4)
    b'\x1aE\xdf\xa3',         # mkv/webm
    b'RIFF',                   # avi
]


def is_video_file(path: Path) -> bool:
    """Return True if file has a video extension OR looks like a video by magic bytes."""
    if path.suffix.lower() in ('.mp4', '.avi', '.mkv', '.mov', '.webm'):
        return True
    # No extension — check magic bytes
    if path.suffix == '':
        try:
            with open(path, 'rb') as f:
                header = f.read(32)
            # ftyp box can appear at offset 4 in mp4
            if b'ftyp' in header[:12]:
                return True
            for magic in VIDEO_MAGIC_BYTES:
                if header.startswith(magic):
                    return True
        except Exception:
            pass
    return False


def video_to_jpg(video_file_path, dst_dir_path):
    """Convert a single video file to JPG frames using ffmpeg."""
    dst_dir_path.mkdir(parents=True, exist_ok=True)

    # Count existing frames (skip if already done)
    existing = list(dst_dir_path.glob("image_*.jpg"))
    if len(existing) > 0:
        print(f"  [skip] {video_file_path.name} — already has {len(existing)} frames")
        return True

    cmd = [
        "ffmpeg",
        "-i", str(video_file_path),
        "-vf", "scale=171:128",        # resize to standard input size
        "-q:v", "1",                   # best quality jpg
        str(dst_dir_path / "image_%05d.jpg"),
        "-hide_banner",
        "-loglevel", "error"           # suppress ffmpeg spam
    ]

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        if result.returncode != 0:
            print(f"  [ERROR] {video_file_path.name}: {result.stderr.decode()}")
            return False
        frame_count = len(list(dst_dir_path.glob("image_*.jpg")))
        print(f"  [done] {video_file_path.name} → {frame_count} frames")
        return True
    except FileNotFoundError:
        print("[FATAL] ffmpeg not found. Install it with: conda install -c conda-forge ffmpeg")
        sys.exit(1)


def convert_dataset(input_dir: Path, output_dir: Path, n_workers: int = 4):
    """
    Walk FYPDATASET structure:
        input_dir/
            fight/         ← class folder
                video1.mp4
                video2.mp4
            Normal/
                ...
            unsafeJump/
                ...
            unsafeThrow/
                ...
            unsafeClimb/
                ...

    Output structure (required by 3D-ResNets):
        output_dir/
            fight/
                video1/
                    image_00001.jpg
                    image_00002.jpg
                    ...
            Normal/
                ...
    """

    class_dirs = [d for d in input_dir.iterdir() if d.is_dir()]

    if not class_dirs:
        print(f"[ERROR] No class subfolders found in: {input_dir}")
        print("Expected: fight/, Normal/, unsafeJump/, unsafeThrow/, unsafeClimb/")
        sys.exit(1)

    print(f"\nFound {len(class_dirs)} class folders: {[d.name for d in class_dirs]}")

    # Collect all (video_path, output_dir) jobs
    # Picks up .mp4/.avi AND extension-less files that are actually videos
    jobs = []
    for class_dir in class_dirs:
        all_files = [f for f in class_dir.iterdir() if f.is_file()]
        videos = [f for f in all_files if is_video_file(f)]
        if not videos:
            print(f"  [warn] No video files found in {class_dir.name}/")
            continue
        no_ext = [v for v in videos if v.suffix == '']
        print(f"  {class_dir.name}: {len(videos)} videos ({len(no_ext)} without extension)")
        for video in sorted(videos):
            # Use stem if has extension, else full name (e.g. "fightclip_03")
            video_stem = video.stem if video.suffix else video.name
            dst = output_dir / class_dir.name / video_stem
            jobs.append((video, dst))

    print(f"Total videos to process: {len(jobs)}\n")

    success = 0
    failed = 0

    # Use threads — safe on Windows (no multiprocessing issues)
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(video_to_jpg, v, d): v for v, d in jobs}
        for future in as_completed(futures):
            ok = future.result()
            if ok:
                success += 1
            else:
                failed += 1

    print(f"\n{'='*50}")
    print(f"Conversion complete.")
    print(f"  Successful : {success}")
    print(f"  Failed     : {failed}")
    print(f"  Output dir : {output_dir}")
    print(f"{'='*50}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Convert FYPDATASET .mp4 videos to JPG frames for 3D-ResNets-PyTorch"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=r"G:/My Drive/DATASET",
        help="Path to your dataset root (contains class subfolders)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=r"G:/My Drive/DATASET_jpg",
        help="Where to save extracted JPG frames"
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)"
    )
    args = parser.parse_args()

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"[ERROR] Input directory not found: {input_dir}")
        print("Make sure Google Drive for Desktop is running and the path is correct.")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Input  : {input_dir}")
    print(f"Output : {output_dir}")

    convert_dataset(input_dir, output_dir, n_workers=args.n_workers)


if __name__ == "__main__":
    main()