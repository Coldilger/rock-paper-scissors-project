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
MODEL_ID = "rock-paper-scissors-sxsw/11"
# MODEL_ID = "hand-detection-e3e9a/12"

INPUT_DIR = Path('data/noaudio_ver0')   # Your source data
OUTPUT_DIR = Path('data/final_dataset') # New clean folder

CONFIDENCE_THRESHOLD = 0.20      # Low threshold to catch early movement
PAD = 30                         # Reduced from 60 to fix "Big Crop"
SPLIT_RATIO = 0.2                # 20% of videos go to Validation

def get_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def process_and_split():
    print(f"Initializing V5 Pipeline (Split + Tight Crop)...")
    
    # 1. Setup Directories
    if OUTPUT_DIR.exists(): shutil.rmtree(OUTPUT_DIR)
    
    (OUTPUT_DIR / 'train').mkdir(parents=True)
    (OUTPUT_DIR / 'val').mkdir(parents=True)
    
    # Initialize Model
    model = get_model(model_id=MODEL_ID, api_key=ROBOFLOW_API_KEY)

    # 2. Iterate Classes
    for class_name in ['Rock', 'Paper', 'Scissor']:
        src_folder = INPUT_DIR / class_name
        if not src_folder.exists(): continue
        
        # Create class folders in train/val
        (OUTPUT_DIR / 'train' / class_name).mkdir(exist_ok=True)
        (OUTPUT_DIR / 'val' / class_name).mkdir(exist_ok=True)
        
        # Group by Video
        pattern = re.compile(r'(.+)_(\d+)\.(jpg|jpeg|png)$', re.IGNORECASE)
        videos = {}
        for img_path in src_folder.glob('*'):
            m = pattern.match(img_path.name)
            if m:
                vid_id = m.group(1)
                frame_idx = int(m.group(2))
                if vid_id not in videos: videos[vid_id] = []
                videos[vid_id].append((frame_idx, img_path))
        
        video_ids = list(videos.keys())
        random.shuffle(video_ids) # Shuffle to split randomly
        
        val_count = int(len(video_ids) * SPLIT_RATIO)
        val_ids = set(video_ids[:val_count])
        
        print(f"Processing {class_name}: {len(video_ids)} videos ({len(val_ids)} validation)")

        # 3. Process Videos
        for vid_id, frames in tqdm(videos.items()):
            frames.sort(key=lambda x: x[0]) 
            
            # Determine destination (Train or Val)
            subset = 'val' if vid_id in val_ids else 'train'
            dest_folder = OUTPUT_DIR / subset / class_name
            
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
                
                # Filter & Select
                candidates = [p for p in predictions if p.confidence > CONFIDENCE_THRESHOLD]
                best_pred = None
                
                if candidates:
                    if last_box is None:
                        # First frame: Closest to center
                        best_pred = min(candidates, key=lambda p: get_distance((p.x, p.y), img_center))
                    else:
                        # Tracking: Closest to last pos
                        last_center = (last_box[0], last_box[1])
                        best_pred = min(candidates, key=lambda p: get_distance((p.x, p.y), last_center))

                # Update State
                current_box = None
                if best_pred:
                    new_box = [best_pred.x, best_pred.y, best_pred.width, best_pred.height]
                    if last_box:
                        vx, vy = new_box[0]-last_box[0], new_box[1]-last_box[1]
                        velocity = [0.6*velocity[0]+0.4*vx, 0.6*velocity[1]+0.4*vy]
                    current_box = new_box
                    last_box = current_box
                    blind_frames = 0
                elif last_box and blind_frames < 5:
                    blind_frames += 1
                    current_box = [last_box[0]+velocity[0], last_box[1]+velocity[1], last_box[2], last_box[3]]
                    last_box = current_box
                else:
                    last_box = None
                    continue # Skip

                # Crop & Save
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

if __name__ == "__main__":
    process_and_split()