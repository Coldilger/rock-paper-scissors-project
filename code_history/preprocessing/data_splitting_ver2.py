import cv2
import os
import re
import math
import shutil
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm
from inference import get_model

# --- CONFIGURATION ---
ROBOFLOW_API_KEY = "K1RsKDud3xFEWD4hdbeu"
MODEL_ID = "hand-detection-e3e9a/12"

INPUT_DIR = Path('data/noaudio_ver2')
OUTPUT_DIR = Path('data/final_split_dataset')

# Tracking Settings
CONFIDENCE_THRESHOLD = 0.4
PAD = 30
MAX_BLIND_FRAMES = 5

# Split Ratios (Must sum to 1.0)
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

def get_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def process_and_split_v6():
    print(f"Initializing V6 Pipeline (Crop + Train/Val/Test Split)...")
    
    # 1. Setup Directories
    if OUTPUT_DIR.exists(): shutil.rmtree(OUTPUT_DIR)
    
    for split in ['train', 'val', 'test']:
        for cls in ['Rock', 'Paper', 'Scissor']:
            (OUTPUT_DIR / split / cls).mkdir(parents=True, exist_ok=True)
    
    try:
        model = get_model(model_id=MODEL_ID, api_key=ROBOFLOW_API_KEY)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 2. Iterate input folders (handling lowercase names)
    # We map 'rock' -> 'Rock', etc. to keep output standard
    class_map = {
        'rock': 'Rock', 'Rock': 'Rock',
        'paper': 'Paper', 'Paper': 'Paper',
        'scissor': 'Scissor', 'scissors': 'Scissor', 'Scissor': 'Scissor'
    }

    for subfolder in INPUT_DIR.iterdir():
        if not subfolder.is_dir(): continue
        
        # Normalize class name
        clean_name = class_map.get(subfolder.name.lower(), None)
        if not clean_name:
            print(f"Skipping unknown folder: {subfolder.name}")
            continue

        # Group by Video
        pattern = re.compile(r'(.+)_(\d+)\.(jpg|jpeg|png)$', re.IGNORECASE)
        videos = {}
        for img_path in subfolder.glob('*'):
            m = pattern.match(img_path.name)
            if m:
                vid_id = m.group(1)
                frame_idx = int(m.group(2))
                if vid_id not in videos: videos[vid_id] = []
                videos[vid_id].append((frame_idx, img_path))
        
        # 3. Create Random Splits
        video_ids = list(videos.keys())
        random.shuffle(video_ids)
        
        n_total = len(video_ids)
        n_train = int(n_total * TRAIN_RATIO)
        n_val = int(n_total * VAL_RATIO)
        
        train_ids = set(video_ids[:n_train])
        val_ids = set(video_ids[n_train : n_train + n_val])
        test_ids = set(video_ids[n_train + n_val:])
        
        print(f"Processing {clean_name}: {n_total} videos "
              f"({len(train_ids)} Train, {len(val_ids)} Val, {len(test_ids)} Test)")

        # 4. Process & Crop
        for vid_id, frames in tqdm(videos.items()):
            frames.sort(key=lambda x: x[0]) 
            
            # Determine destination
            if vid_id in train_ids:   subset = 'train'
            elif vid_id in val_ids:   subset = 'val'
            else:                     subset = 'test'
            
            dest_folder = OUTPUT_DIR / subset / clean_name
            
            # Tracking State
            last_box = None
            velocity = [0, 0]
            blind_frames = 0
            
            for idx, (frame_num, img_path) in enumerate(frames):
                image = cv2.imread(str(img_path))
                if image is None: continue
                h_img, w_img, _ = image.shape
                img_center = (w_img / 2, h_img / 2)
                
                results = model.infer(image)
                predictions = results[0].predictions if isinstance(results, list) else results.predictions
                
                candidates = [p for p in predictions if p.confidence > CONFIDENCE_THRESHOLD]
                best_pred = None
                
                if candidates:
                    if last_box is None:
                        # First frame: Closest to center
                        best_pred = min(candidates, key=lambda p: get_distance((p.x, p.y), img_center))
                        # Center sanity check
                        if get_distance((best_pred.x, best_pred.y), img_center) > (w_img * 0.4):
                            best_pred = None 
                    else:
                        # Tracking: Closest to last pos
                        last_center = (last_box[0], last_box[1])
                        best_pred = min(candidates, key=lambda p: get_distance((p.x, p.y), last_center))

                current_box = None
                if best_pred:
                    new_box = [best_pred.x, best_pred.y, best_pred.width, best_pred.height]
                    if last_box:
                        vx, vy = new_box[0]-last_box[0], new_box[1]-last_box[1]
                        if blind_frames > 0: velocity = [vx, vy]
                        else: velocity = [0.6*velocity[0]+0.4*vx, 0.6*velocity[1]+0.4*vy]
                    current_box = new_box
                    last_box = current_box
                    blind_frames = 0
                elif last_box and blind_frames < MAX_BLIND_FRAMES:
                    blind_frames += 1
                    current_box = [last_box[0]+velocity[0], last_box[1]+velocity[1], last_box[2], last_box[3]]
                    last_box = current_box
                else:
                    last_box = None
                    continue 

                if current_box:
                    cx, cy, w, h = current_box
                    
                    x1 = int(cx - w/2) - PAD
                    y1 = int(cy - h/2) - PAD
                    x2 = int(cx + w/2) + PAD
                    y2 = int(cy + h/2) + PAD
                    
                    x1 = max(0, x1); y1 = max(0, y1)
                    x2 = min(w_img, x2); y2 = min(h_img, y2)
                    
                    crop = image[y1:y2, x1:x2]
                    if crop.size > 0:
                        cv2.imwrite(str(dest_folder / img_path.name), crop)

    print(f"\nDone! Dataset split saved to: {OUTPUT_DIR}")
    print(f"  - Train: Use for model.fit()")
    print(f"  - Val:   Use for early stopping/saving best model")
    print(f"  - Test:  Use for final 'Minority Report' graph evaluation")

if __name__ == "__main__":
    process_and_split_v6()