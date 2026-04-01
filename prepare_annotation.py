import json
import random
import argparse
import sys
from pathlib import Path


# ── Config ────────────────────────────────────────────────────────────────────
CLASSES     = ['fight', 'Normal', 'unsafeJump', 'unsafeThrow', 'unsafeClimb']
TRAIN_RATIO = 0.8
RANDOM_SEED = 42
# ─────────────────────────────────────────────────────────────────────────────


def count_frames(video_dir: Path) -> int:
    """Count image_*.jpg files in a video folder."""
    return len(list(video_dir.glob("image_*.jpg")))


def build_database(jpg_root: Path, train_ratio: float):
    """
    Walk DATASET_jpg and build the annotation database.

    Expected structure:
        jpg_root/
            fight/
                video1/
                    image_00001.jpg ...
            Normal/
                video2/ ...
            unsafeJump/ ...
            unsafeThrow/ ...
            unsafeClimb/ ...
    """
    random.seed(RANDOM_SEED)

    database    = {}
    labels      = []
    skipped     = []
    train_count = 0
    val_count   = 0

    for class_name in CLASSES:
        class_dir = jpg_root / class_name

        if not class_dir.exists():
            print(f"  [warn] Class folder not found, skipping: {class_dir}")
            continue

        # Each subfolder = one video
        video_dirs = sorted([d for d in class_dir.iterdir() if d.is_dir()])

        if not video_dirs:
            print(f"  [warn] No video subfolders in {class_name}/")
            continue

        labels.append(class_name)

        # Shuffle then split 80/20
        random.shuffle(video_dirs)
        split_idx  = max(1, int(len(video_dirs) * train_ratio))
        train_vids = set(video_dirs[:split_idx])
        val_vids   = set(video_dirs[split_idx:])

        print(f"  {class_name}: {len(video_dirs)} videos "
              f"-> {len(train_vids)} train / {len(val_vids)} val")

        for video_dir in video_dirs:
            n_frames = count_frames(video_dir)

            if n_frames == 0:
                print(f"    [skip] {video_dir.name} - no jpg frames found")
                skipped.append(str(video_dir))
                continue

            subset = "training" if video_dir in train_vids else "validation"

            database[video_dir.name] = {
                "subset": subset,
                "annotations": {
                    "label":   class_name,
                    "segment": [1, n_frames]
                }
            }

            if subset == "training":
                train_count += 1
            else:
                val_count += 1

    return database, labels, train_count, val_count, skipped


def main():
    parser = argparse.ArgumentParser(
        description="Generate annotation JSON for 3D-ResNets-PyTorch (custom dataset)"
    )
    parser.add_argument(
        "--jpg_dir",
        type=str,
        default=r"G:/My Drive/DATASET_jpg",
        help="Path to extracted JPG frames root (output of generate_video_jpgs.py)"
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=r"G:/My Drive/DATASET_jpg/dataset.json",
        help="Where to save the annotation JSON file"
    )
    args = parser.parse_args()

    jpg_root    = Path(args.jpg_dir)
    output_json = Path(args.output_json)

    if not jpg_root.exists():
        print(f"[ERROR] JPG directory not found: {jpg_root}")
        print("Make sure generate_video_jpgs.py ran successfully first.")
        sys.exit(1)

    print(f"\nReading frames from : {jpg_root}")
    print(f"Output JSON         : {output_json}\n")

    database, labels, train_count, val_count, skipped = build_database(
        jpg_root, TRAIN_RATIO
    )

    if not database:
        print("\n[ERROR] No videos found. Check your DATASET_jpg folder structure.")
        sys.exit(1)

    annotation = {
        "labels":   labels,
        "database": database
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(annotation, f, indent=2)

    print(f"\n{'='*50}")
    print(f"JSON created successfully!")
    print(f"  Classes    : {labels}")
    print(f"  Training   : {train_count} videos")
    print(f"  Validation : {val_count} videos")
    print(f"  Total      : {train_count + val_count} videos")
    if skipped:
        print(f"  Skipped    : {len(skipped)} (no frames found)")
    print(f"  Saved to   : {output_json}")
    print(f"{'='*50}\n")

    print("Next step - run training with:")
    print(f"""
python main.py ^
  --root_path "G:/My Drive/DATASET_jpg" ^
  --video_path . ^
  --annotation_path dataset.json ^
  --result_path results ^
  --dataset ucf101 ^
  --model resnet ^
  --model_depth 50 ^
  --n_classes {len(labels)} ^
  --batch_size 32 ^
  --n_threads 4 ^
  --checkpoint 5
""")


if __name__ == "__main__":
    main()