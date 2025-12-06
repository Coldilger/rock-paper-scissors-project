import cv2
import torch
import torch.nn as nn
import numpy as np
import torchvision.models as models
import time
import os
from collections import deque
from inference import get_model
from torchvision import transforms
from PIL import Image

# --- CONFIGURATION ---
ROBOFLOW_API_KEY = "K1RsKDud3xFEWD4hdbeu"
MODEL_ID = "hand-detection-e3e9a/12" # Generic Hand Detector

# Path handling (same folder as script)
script_dir = os.path.dirname(os.path.abspath(__file__))
TCN_MODEL_PATH = os.path.join(script_dir, "rps_tcn_model.pth") 

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Updated Hyperparameters (Must match your Training Code!)
SEQ_LEN = 64
FRAME_SIZE = 224
FEATURE_DIM = 128 
CHANNELS = [64, 64, 64, 64]
KERNEL_SIZE = 5

CLASS_MAP = {0: 'Rock', 1: 'Paper', 2: 'Scissor'}
WINNING_MOVE = {'Rock': 'Paper', 'Paper': 'Scissor', 'Scissor': 'Rock'}

# --- MODEL CLASSES (MUST MATCH TRAINING EXACTLY) ---
class ResNetFrameEncoder(nn.Module):
    def __init__(self, feature_dim=64):
        super().__init__()
        resnet = models.resnet18(weights=None) 
        modules = list(resnet.children())[:-1] 
        self.backbone = nn.Sequential(*modules)
        self.proj = nn.Linear(512, feature_dim)

    def forward(self, x):
        features = self.backbone(x)
        features = features.flatten(1)
        return self.proj(features)

class GestureTCN(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.encoder = ResNetFrameEncoder(feature_dim=FEATURE_DIM) 
        from pytorch_tcn import TCN
        self.tcn = TCN(
            num_inputs=FEATURE_DIM, 
            num_channels=CHANNELS, 
            kernel_size=KERNEL_SIZE,
            dropout=0.2, 
            causal=True, 
            input_shape='NCL' 
        )
        self.classifier = nn.Linear(CHANNELS[-1], num_classes)

    def forward(self, x):
        b, t, c, h, w = x.shape
        x_flat = x.view(b * t, c, h, w)
        features = self.encoder(x_flat)
        features = features.view(b, t, -1).permute(0, 2, 1)
        tcn_out = self.tcn(features)
        last_out = tcn_out[:, :, -1]
        return self.classifier(last_out)

# --- UTILS ---
transform = transforms.Compose([
    transforms.Resize((FRAME_SIZE, FRAME_SIZE)),
    transforms.ToTensor(),
])

def run_game():
    # 1. Load Models
    print("Loading Detector...")
    detector = get_model(model_id=MODEL_ID, api_key=ROBOFLOW_API_KEY)
    
    print(f"Loading TCN Brain from {TCN_MODEL_PATH}...")
    model = GestureTCN(num_classes=3).to(DEVICE)
    try:
        model.load_state_dict(torch.load(TCN_MODEL_PATH, map_location=DEVICE))
    except RuntimeError as e:
        print("\nERROR: Model size mismatch! Did you retrain with FEATURE_DIM=128 and CHANNELS=[64,64,64,64]?")
        print(e)
        return
    model.eval()
    
    # 2. Setup Camera
    cap = cv2.VideoCapture(0)
    
    # Game State Variables
    state = "IDLE" # IDLE, COUNTDOWN, RECORDING, RESULT
    start_time = 0
    buffer = [] # List to store frames for TCN
    
    # Lag Fix Variables
    frame_count = 0
    SKIP_FRAMES = 4 # Run detector only every 4th frame
    last_known_box = None
    
    final_result_text = ""
    
    print("Starting Battle! Press 'SPACE' to start a round. 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # Mirror for better UX
        frame = cv2.flip(frame, 1)
        h_img, w_img, _ = frame.shape
        
        # --- A. FAST DETECTION (SKIP FRAMES) ---
        current_box = None
        
        if frame_count % SKIP_FRAMES == 0:
            # Run Heavy Inference
            results = detector.infer(frame)
            predictions = results[0].predictions if isinstance(results, list) else results.predictions
            
            # Find best box
            best_pred = None
            max_conf = -1
            for pred in predictions:
                if pred.confidence > 0.3 and pred.confidence > max_conf:
                    max_conf = pred.confidence
                    best_pred = pred
            
            if best_pred:
                # Save as (x1, y1, x2, y2)
                x, y, w, h = best_pred.x, best_pred.y, best_pred.width, best_pred.height
                x1 = int(x - w/2); y1 = int(y - h/2)
                x2 = int(x + w/2); y2 = int(y + h/2)
                
                # Add Padding (Match your training pad!)
                pad = 30
                x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
                x2 = min(w_img, x2 + pad); y2 = min(h_img, y2 + pad)
                
                last_known_box = (x1, y1, x2, y2)
                current_box = last_known_box
            else:
                last_known_box = None # Lost hand
        else:
            # FAST PATH: Reuse last box
            current_box = last_known_box
            
        frame_count += 1

        # --- B. GAME LOGIC ---
        ui_text = ""
        ui_color = (255, 255, 255)
        
        # Draw Box if found
        if current_box:
            cv2.rectangle(frame, (current_box[0], current_box[1]), (current_box[2], current_box[3]), (0, 255, 0), 2)

        if state == "IDLE":
            ui_text = "Press SPACE to Start"
            buffer = [] # Reset buffer
            
        elif state == "COUNTDOWN":
            elapsed = time.time() - start_time
            if elapsed < 1.0:
                ui_text = "1..."
            elif elapsed < 2.0:
                ui_text = "2..."
            else:
                state = "RECORDING"
                
        elif state == "RECORDING":
            ui_text = "GO! Throw!"
            ui_color = (0, 255, 255) # Yellow
            
            # We need a crop to record
            if current_box:
                x1, y1, x2, y2 = current_box
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                    tensor_img = transform(pil_img)
                    buffer.append(tensor_img)
            
            # Check if done
            if len(buffer) >= SEQ_LEN:
                state = "PREDICT"
                
        elif state == "PREDICT":
            # Run Inference ONE TIME per round
            seq = torch.stack(buffer).unsqueeze(0).to(DEVICE) # (1, 64, 3, 224, 224)
            
            with torch.no_grad():
                logits = model(seq)
                probs = torch.softmax(logits, dim=1)
                conf, pred_idx = torch.max(probs, 1)
                
                user_move = CLASS_MAP[pred_idx.item()]
                my_move = WINNING_MOVE[user_move]
                
                final_result_text = f"You: {user_move} | I Play: {my_move}"
                print(f"Game Over. {final_result_text} (Conf: {conf.item():.2f})")
                
                state = "RESULT"
                start_time = time.time() # Start cooldown timer

        elif state == "RESULT":
            ui_text = final_result_text
            ui_color = (0, 0, 255) # Red
            
            # Show result for 4 seconds then reset
            if time.time() - start_time > 4.0:
                state = "IDLE"

        # --- C. DRAW UI ---
        # Add a black background bar for text legibility
        cv2.rectangle(frame, (0, 0), (w_img, 80), (0, 0, 0), -1)
        cv2.putText(frame, ui_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, ui_color, 3)
        
        cv2.imshow("Rock Paper Scissors AI", frame)
        
        # Input Handling
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord(' ') and state == "IDLE":
            state = "COUNTDOWN"
            start_time = time.time()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_game()