import cv2
import os
import re
import numpy as np
from pathlib import Path
from tqdm import tqdm
from inference import get_model

# --- CONFIGURATION ---
ROBOFLOW_API_KEY = "K1RsKDud3xFEWD4hdbeu"
MODEL_ID = "hand-detection-6hbu3/1"
# MODEL_ID = "rock-paper-scissors-sxsw/11"
INPUT_DIR = Path('data/noaudio_ver1')
OUTPUT_DIR = Path('data/noaudio_ver1_tracking') # New folder
CONFIDENCE_THRESHOLD = 0.3 # Lower threshold slightly to catch blurry hands

def crop_dataset_tracking():
    print(f"Initializing Roboflow model...")
    model = get_model(model_id=MODEL_ID, api_key=ROBOFLOW_API_KEY)
    
    if not OUTPUT_DIR.exists():
        OUTPUT_DIR.mkdir(parents=True)
    
    # Process by Class
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
            
            # State variables for tracking
            last_box = None # [x_center, y_center, w, h]
            velocity = [0, 0] # [vx, vy] (Change in X and Y per frame)
            
            for idx, (frame_num, img_path) in enumerate(frames):
                image = cv2.imread(str(img_path))
                if image is None: continue
                h_img, w_img, _ = image.shape
                
                # Inference
                results = model.infer(image)
                predictions = results[0].predictions if isinstance(results, list) else results.predictions
                
                best_pred = None
                max_conf = -1
                for pred in predictions:
                    if pred.confidence > max_conf:
                        max_conf = pred.confidence
                        best_pred = pred
                
                # --- TRACKING LOGIC ---
                current_box = None
                
                # 1. DETECTED: Update Box and Calculate Velocity
                if best_pred and max_conf > CONFIDENCE_THRESHOLD:
                    new_box = [best_pred.x, best_pred.y, best_pred.width, best_pred.height]
                    
                    if last_box is not None:
                        # Calculate how much it moved since last frame
                        vx = new_box[0] - last_box[0]
                        vy = new_box[1] - last_box[1]
                        
                        # Simple smoothing (optional, keeps movement less jittery)
                        velocity = [0.7*velocity[0] + 0.3*vx, 0.7*velocity[1] + 0.3*vy]
                    
                    current_box = new_box
                    last_box = current_box

                # 2. NOT DETECTED: Predict Position using Velocity
                elif last_box is not None:
                    # Apply velocity to guess next position
                    predicted_x = last_box[0] + velocity[0]
                    predicted_y = last_box[1] + velocity[1]
                    
                    # Keep W and H the same
                    current_box = [predicted_x, predicted_y, last_box[2], last_box[3]]
                    
                    # Update state so next frame continues from here
                    last_box = current_box
                    # We keep velocity same (linear extrapolation) or decay it slightly
                    # velocity = [v * 0.9 for v in velocity] 

                # --- CROPPING ---
                if current_box:
                    cx, cy, w, h = current_box
                    
                    # Add generous padding (margin)
                    # This is CRITICAL. Even if our prediction is slightly off, 
                    # extra padding ensures the hand is still in the frame.
                    pad = 40 
                    
                    x1 = int(cx - w/2) - pad
                    y1 = int(cy - h/2) - pad
                    x2 = int(cx + w/2) + pad
                    y2 = int(cy + h/2) + pad
                    
                    # Clamp to image bounds
                    x1 = max(0, x1); y1 = max(0, y1)
                    x2 = min(w_img, x2); y2 = min(h_img, y2)
                    
                    crop = image[y1:y2, x1:x2]
                    
                    if crop.size > 0:
                        save_path = OUTPUT_DIR / subfolder.name / img_path.name
                        cv2.imwrite(str(save_path), crop)

if __name__ == "__main__":
    crop_dataset_tracking()