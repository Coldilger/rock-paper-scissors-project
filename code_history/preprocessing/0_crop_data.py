import cv2
import os
import re
from pathlib import Path
from tqdm import tqdm
from inference import get_model
import numpy as np

# --- CONFIGURATION ---
ROBOFLOW_API_KEY = "K1RsKDud3xFEWD4hdbeu"  # Your Key
MODEL_ID = "rock-paper-scissors-sxsw/11"   # The Object Detection Model
CONFIDENCE_THRESHOLD = 0.4                 # Minimum confidence to accept a hand

# Paths
INPUT_DIR = Path('data/noaudio_ver2') #==========================================change if needed
OUTPUT_DIR = Path('data/noaudio_ver2_cropped')#==================================change if needed

def crop_dataset_smart():
    print(f"Initializing Roboflow model...")
    model = get_model(model_id=MODEL_ID, api_key=ROBOFLOW_API_KEY)
    
    if not OUTPUT_DIR.exists():
        OUTPUT_DIR.mkdir(parents=True)
    
    # Process by Class (Rock/Paper/Scissor)
    for subfolder in INPUT_DIR.iterdir():
        if not subfolder.is_dir(): continue
        
        # Create output subfolder
        (OUTPUT_DIR / subfolder.name).mkdir(exist_ok=True)
        
        # Group files by Video ID so we can process them sequentially
        # Regex to capture "VideoID" from "VideoID_FrameNumber.jpg"
        pattern = re.compile(r'(.+)_(\d+)\.(jpg|jpeg|png)$', re.IGNORECASE)
        videos = {}
        
        # 1. Group images by Video
        for img_path in subfolder.glob('*'):
            m = pattern.match(img_path.name)
            if m:
                vid_id = m.group(1)
                frame_idx = int(m.group(2))
                if vid_id not in videos: videos[vid_id] = []
                videos[vid_id].append((frame_idx, img_path))
        
        print(f"Processing {subfolder.name}: {len(videos)} videos found.")

        # 2. Process each video sequentially
        for vid_id, frames in tqdm(videos.items()):
            frames.sort(key=lambda x: x[0]) # Sort by frame number (Crucial!)
            
            last_known_box = None # (x, y, w, h)
            
            for idx, (frame_num, img_path) in enumerate(frames):
                image = cv2.imread(str(img_path))
                if image is None: continue
                h_img, w_img, _ = image.shape
                
                # Run Inference
                results = model.infer(image)
                predictions = results[0].predictions if isinstance(results, list) else results.predictions
                
                # Find best hand
                best_pred = None
                max_conf = -1
                for pred in predictions:
                    if pred.confidence > max_conf:
                        max_conf = pred.confidence
                        best_pred = pred
                
                # DECISION LOGIC:
                current_box = None
                
                # Case A: Good detection found
                if best_pred and max_conf > CONFIDENCE_THRESHOLD:
                    current_box = (best_pred.x, best_pred.y, best_pred.width, best_pred.height)
                    last_known_box = current_box # Update memory
                
                # Case B: Bad detection, but we have history (Motion Blur Handling)
                elif last_known_box is not None:
                    # We assume the hand didn't teleport. Use the last known position.
                    current_box = last_known_box
                
                # If we have a box (either new or memorized), Crop & Save
                if current_box:
                    x, y, w, h = current_box
                    x1 = int(x - w/2) - 20
                    y1 = int(y - h/2) - 20
                    x2 = int(x + w/2) + 20
                    y2 = int(y + h/2) + 20
                    
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w_img, x2), min(h_img, y2)
                    
                    crop = image[y1:y2, x1:x2]
                    
                    save_path = OUTPUT_DIR / subfolder.name / img_path.name
                    if crop.size > 0:
                        cv2.imwrite(str(save_path), crop)

if __name__ == "__main__":
    crop_dataset_smart()