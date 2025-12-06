import cv2
import os
import re
import math
import shutil
import numpy as np
from pathlib import Path
from tqdm import tqdm
from inference import get_model

# --- CONFIGURATION ---
ROBOFLOW_API_KEY = "K1RsKDud3xFEWD4hdbeu"
MODEL_ID = "hand-detection-e3e9a/12"

INPUT_DIR = Path('data/noaudio_ver1')
OUTPUT_DIR = Path('data/noaudio_ver1_tracking') # We will OVERWRITE this

# Strictness settings
CONFIDENCE_THRESHOLD = 0.25      # Minimum confidence to start tracking
MAX_FRAMES_TO_TRACK_BLIND = 5    # How many frames to "guess" if hand disappears before giving up

def get_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def crop_dataset_v4():
    print(f"Initializing V4 Pipeline (Smart Drop + Tracking)...")
    
    # 1. CLEANUP: Delete old output folder if it exists
    if OUTPUT_DIR.exists():
        print(f"Removing existing folder {OUTPUT_DIR}...")
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)

    try:
        model = get_model(model_id=MODEL_ID, api_key=ROBOFLOW_API_KEY)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Process subfolders
    for subfolder in INPUT_DIR.iterdir():
        if not subfolder.is_dir(): continue
        (OUTPUT_DIR / subfolder.name).mkdir(exist_ok=True)
        
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
        
        print(f"Processing {subfolder.name}: {len(videos)} videos found.")

        for vid_id, frames in tqdm(videos.items()):
            frames.sort(key=lambda x: x[0]) 
            
            # Reset Tracking State for new video
            last_box = None     # [cx, cy, w, h]
            velocity = [0, 0]   # [vx, vy]
            blind_frames = 0    # Counter for how long we've been guessing
            
            for idx, (frame_num, img_path) in enumerate(frames):
                image = cv2.imread(str(img_path))
                if image is None: continue
                h_img, w_img, _ = image.shape
                img_center = (w_img / 2, h_img / 2)
                
                # Inference
                results = model.infer(image)
                predictions = results[0].predictions if isinstance(results, list) else results.predictions
                
                # Filter candidates
                candidates = [p for p in predictions if p.confidence > CONFIDENCE_THRESHOLD]
                
                best_pred = None
                
                # --- SELECTION LOGIC ---
                if candidates:
                    if last_box is None:
                        # FIRST DETECTION: Must be reasonably central
                        best_pred = min(candidates, key=lambda p: get_distance((p.x, p.y), img_center))
                        # If too far from center (likely garbage background), ignore
                        if get_distance((best_pred.x, best_pred.y), img_center) > (w_img * 0.35):
                            best_pred = None 
                    else:
                        # TRACKING: Pick candidate closest to last known position
                        last_center = (last_box[0], last_box[1])
                        best_pred = min(candidates, key=lambda p: get_distance((p.x, p.y), last_center))

                # --- UPDATE TRACKER ---
                current_box = None
                
                if best_pred:
                    # CASE A: We found the hand!
                    new_box = [best_pred.x, best_pred.y, best_pred.width, best_pred.height]
                    
                    if last_box is not None:
                        # Update velocity
                        vx = new_box[0] - last_box[0]
                        vy = new_box[1] - last_box[1]
                        # Smooth velocity update
                        if blind_frames > 0:
                            velocity = [vx, vy] # Reset if we just recovered
                        else:
                            velocity = [0.6*velocity[0] + 0.4*vx, 0.6*velocity[1] + 0.4*vy]
                    
                    current_box = new_box
                    last_box = current_box
                    blind_frames = 0 # Reset blind counter

                elif last_box is not None and blind_frames < MAX_FRAMES_TO_TRACK_BLIND:
                    # CASE B: Hand lost, but we "guess" using velocity
                    blind_frames += 1
                    
                    pred_x = last_box[0] + velocity[0]
                    pred_y = last_box[1] + velocity[1]
                    current_box = [pred_x, pred_y, last_box[2], last_box[3]]
                    
                    last_box = current_box
                    # Don't update velocity while guessing
                
                else:
                    # CASE C: No hand found AND we ran out of blind frames.
                    # This frame is DROPPED.
                    last_box = None # Lost track completely
                    velocity = [0, 0]
                    continue # Skip saving

                # --- CROP & SAVE ---
                if current_box:
                    cx, cy, w, h = current_box
                    pad = 30 # Extra padding for safety=======================================
                    
                    x1 = int(cx - w/2) - pad
                    y1 = int(cy - h/2) - pad
                    x2 = int(cx + w/2) + pad
                    y2 = int(cy + h/2) + pad
                    
                    x1 = max(0, x1); y1 = max(0, y1)
                    x2 = min(w_img, x2); y2 = min(h_img, y2)
                    
                    crop = image[y1:y2, x1:x2]
                    
                    if crop.size > 0:
                        save_path = OUTPUT_DIR / subfolder.name / img_path.name
                        cv2.imwrite(str(save_path), crop)

    print(f"Done! Cleaned dataset saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    crop_dataset_v4()