import re
import math
import random
import shutil
from pathlib import Path

# ------------- CONFIG -------------
# Where your extracted frames currently are:
# e.g. "videos_noaudio" or "videos_transformed"
SOURCE_ROOT = Path(r"code\videos_noaudio")   # <-- change if needed

# Where to build the dataset structure:
DEST_ROOT = Path("data/final_split_dataset")

CLASSES = ["Rock", "Paper", "Scissor"]

# Split ratios (by VIDEO, not by frame)
TRAIN_RATIO = 0.7
VAL_RATIO   = 0.15   # rest goes to test

# Copy or move frames?
MOVE_FILES = False    # set to True if you want to move instead of copy
# ----------------------------------


def copy_or_move(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if MOVE_FILES:
        shutil.move(str(src), str(dst))
    else:
        shutil.copy2(str(src), str(dst))


def main():
    # Regex: (video_id)_(frame_number).jpg
    # Matches e.g. "paper00_720p20fps_001.jpg"
    pattern = re.compile(r"(.+)_([0-9]+)\.(jpg|jpeg|png)$", re.IGNORECASE)

    for cls in CLASSES:
        src_dir = SOURCE_ROOT / cls
        if not src_dir.exists():
            print(f"⚠️ Source class folder not found: {src_dir}")
            continue

        print(f"\n=== Processing class: {cls} ===")
        # Group frames by video id
        videos = {}  # video_id -> list[Path]

        for img_path in src_dir.glob("*"):
            if not img_path.is_file():
                continue

            m = pattern.match(img_path.name)
            if not m:
                # Skip non-frame files (mp4 etc.)
                continue

            video_id = m.group(1)   # everything before last _###.ext
            videos.setdefault(video_id, []).append(img_path)

        if not videos:
            print(f"⚠️ No frame images found in {src_dir}")
            continue

        # Sort frames inside each video by index
        for vid, frames in videos.items():
            videos[vid] = sorted(
                frames,
                key=lambda p: int(pattern.match(p.name).group(2))
            )

        # Split video IDs into train / val / test
        video_ids = list(videos.keys())
        random.shuffle(video_ids)

        n = len(video_ids)
        n_train = math.floor(n * TRAIN_RATIO)
        n_val   = math.floor(n * VAL_RATIO)

        train_ids = video_ids[:n_train]
        val_ids   = video_ids[n_train:n_train + n_val]
        test_ids  = video_ids[n_train + n_val:]

        print(f"Found {n} videos in class {cls}: "
              f"{len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test")

        # Helper to copy a whole video’s frames to a split
        def handle_split(id_list, split_name):
            for vid in id_list:
                frames = videos[vid]
                for src_frame in frames:
                    dst_dir = DEST_ROOT / split_name / cls
                    dst_frame = dst_dir / src_frame.name
                    copy_or_move(src_frame, dst_frame)

        handle_split(train_ids, "train")
        handle_split(val_ids,   "val")
        handle_split(test_ids,  "test")

    print("\n✅ Done building data/final_split_dataset/")
    print("   Structure:")
    print("   data/final_split_dataset/train|val|test/Rock|Paper|Scissor")


if __name__ == "__main__":
    main()
