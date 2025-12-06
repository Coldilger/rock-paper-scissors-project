import cv2
import os
import re
import math
import numpy as np
from pathlib import Path
from tqdm import tqdm
from inference import get_model

# --- CONFIGURATION ---
ROBOFLOW_API_KEY = "K1RsKDud3xFEWD4hdbeu"
MODEL_ID = "hand-detection-e3e9a/12"  # Keeping the generic hand detector

INPUT_DIR = Path('data/noaudio_ver1')
OUTPUT_DIR = Path('data/noaudio_ver1_tracking') # New folder
CONFIDENCE_THRESHOLD = 0.20 # Lowered significantly! We rely on LOCATION now, not just confidence.

def get_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def crop_dataset_center_lock():
    print(f"Initializing Hand Detector ({MODEL_ID}) with Center-Lock Logic...")
    model = get_model(model_id=MODEL_ID, api_key=ROBOFLOW_API_KEY)
    
    if not OUTPUT_DIR.exists():
        OUTPUT_DIR.mkdir(parents=True)
    
    for subfolder in INPUT_DIR.iterdir():
        if not subfolder.is_dir(): continue
        (OUTPUT_DIR / subfolder.name).mkdir(exist_ok=True)
        
        # Group by Video ID
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
            
            last_box = None     # [cx, cy, w, h]
            velocity = [0, 0]   # [vx, vy]
            frames_lost_count = 0 
            
            for idx, (frame_num, img_path) in enumerate(frames):
                image = cv2.imread(str(img_path))
                if image is None: continue
                h_img, w_img, _ = image.shape
                img_center = (w_img / 2, h_img / 2)
                
                # Inference
                results = model.infer(image)
                predictions = results[0].predictions if isinstance(results, list) else results.predictions
                
                # --- SMARTER SELECTION LOGIC ---
                best_pred = None
                
                # 1. Filter valid candidates first
                candidates = [p for p in predictions if p.confidence > CONFIDENCE_THRESHOLD]
                
                if candidates:
                    if last_box is None:
                        # FIRST FRAME: Pick the candidate closest to the IMAGE CENTER
                        # This kills the "Corner Ghost" issue
                        best_pred = min(candidates, key=lambda p: get_distance((p.x, p.y), img_center))
                        
                        # Sanity check: If the "closest" is still wildly far (like > 40% of screen away), ignore it
                        dist = get_distance((best_pred.x, best_pred.y), img_center)
                        if dist > (w_img * 0.4): 
                            best_pred = None # Too far from center, probably garbage
                            
                    else:
                        # SUBSEQUENT FRAMES: Pick candidate closest to LAST KNOWN POSITION
                        last_center = (last_box[0], last_box[1])
                        best_pred = min(candidates, key=lambda p: get_distance((p.x, p.y), last_center))

                # --- TRACKING UPDATE (Same as before) ---
                current_box = None
                
                if best_pred:
                    new_box = [best_pred.x, best_pred.y, best_pred.width, best_pred.height]
                    
                    if last_box is not None:
                        vx = new_box[0] - last_box[0]
                        vy = new_box[1] - last_box[1]
                        if frames_lost_count > 0:
                            velocity = [vx, vy]
                        else:
                            velocity = [0.6*velocity[0] + 0.4*vx, 0.6*velocity[1] + 0.4*vy]
                    
                    current_box = new_box
                    last_box = current_box
                    frames_lost_count = 0 

                elif last_box is not None:
                    # Physics Fallback
                    frames_lost_count += 1
                    pred_x = last_box[0] + velocity[0]
                    pred_y = last_box[1] + velocity[1]
                    current_box = [pred_x, pred_y, last_box[2], last_box[3]]
                    last_box = current_box

                # --- CROP ---
                if current_box:
                    cx, cy, w, h = current_box
                    pad = 50 
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

if __name__ == "__main__":
    crop_dataset_center_lock()