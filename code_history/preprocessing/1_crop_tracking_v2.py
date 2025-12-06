import cv2
import os
import re
import numpy as np
from pathlib import Path
from tqdm import tqdm
from inference import get_model

# --- CONFIGURATION ---
ROBOFLOW_API_KEY = "K1RsKDud3xFEWD4hdbeu"
MODEL_ID = "hand-detection-e3e9a/12"
# MODEL_ID = "rock-paper-scissors-sxsw/11"
INPUT_DIR = Path('data/noaudio_ver2')
OUTPUT_DIR = Path('data/noaudio_ver2_tracking') # New folder
CONFIDENCE_THRESHOLD = 0.4

def crop_dataset_hand_tracking():
    print(f"Initializing Generic Hand Detector ({MODEL_ID})...")
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
            
            # --- TRACKING STATE ---
            last_box = None     # [cx, cy, w, h]
            velocity = [0, 0]   # [vx, vy]
            frames_lost_count = 0 
            
            for idx, (frame_num, img_path) in enumerate(frames):
                image = cv2.imread(str(img_path))
                if image is None: continue
                h_img, w_img, _ = image.shape
                
                # Inference
                results = model.infer(image)
                # Handle list vs object return type safety
                predictions = results[0].predictions if isinstance(results, list) else results.predictions
                
                # Find best Hand (highest confidence)
                best_pred = None
                max_conf = -1
                for pred in predictions:
                    if pred.confidence > max_conf:
                        max_conf = pred.confidence
                        best_pred = pred
                
                current_box = None
                
                # --- LOGIC A: Hand Detected ---
                if best_pred and max_conf > CONFIDENCE_THRESHOLD:
                    new_box = [best_pred.x, best_pred.y, best_pred.width, best_pred.height]
                    
                    if last_box is not None:
                        # Calculate instantaneous velocity
                        vx = new_box[0] - last_box[0]
                        vy = new_box[1] - last_box[1]
                        
                        # Update velocity with smoothing
                        # If we just found the hand after losing it, trust the new detection more
                        if frames_lost_count > 0:
                            velocity = [vx, vy]
                        else:
                            velocity = [0.6*velocity[0] + 0.4*vx, 0.6*velocity[1] + 0.4*vy]
                    
                    current_box = new_box
                    last_box = current_box
                    frames_lost_count = 0 # Reset counter

                # --- LOGIC B: Hand Lost (Use Physics) ---
                elif last_box is not None:
                    frames_lost_count += 1
                    
                    # Apply momentum
                    pred_x = last_box[0] + velocity[0]
                    pred_y = last_box[1] + velocity[1]
                    
                    # Assume size stays roughly the same
                    current_box = [pred_x, pred_y, last_box[2], last_box[3]]
                    
                    # Update "last_box" to this predicted position so the next frame 
                    # continues from here (chain reaction)
                    last_box = current_box
                    
                    # Do NOT decay velocity to zero immediately. 
                    # Hands in freefall (gestures) don't stop mid-air.
                    # We keep velocity constant for a few frames.

                # --- CROP ---
                if current_box:
                    cx, cy, w, h = current_box
                    
                    # Generous Padding
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
    crop_dataset_hand_tracking()