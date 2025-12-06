import cv2
import torch
import torch.nn as nn
import numpy as np
import torchvision.models as models
import time
import os
import mediapipe as mp
from collections import deque
from torchvision import transforms
from PIL import Image

# --- CONFIGURATION ---
# Auto-detect path to model file (must be in same folder as this script)
script_dir = os.path.dirname(os.path.abspath(__file__))
TCN_MODEL_PATH = os.path.join(script_dir, "rps_tcn_model.pth") 

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters (Must match your Training!)
SEQ_LEN = 64
FRAME_SIZE = 224
FEATURE_DIM = 128
CHANNELS = [64, 64, 64, 64]
KERNEL_SIZE = 6
PAD = 50 

CLASS_MAP = {0: 'Rock', 1: 'Paper', 2: 'Scissor'}
WINNING_MOVE = {'Rock': 'Paper', 'Paper': 'Scissor', 'Scissor': 'Rock'}

# --- MODEL DEFINITIONS ---
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
    # 1. Load TCN
    print(f"Loading TCN Brain from {TCN_MODEL_PATH}...")
    model = GestureTCN(num_classes=3).to(DEVICE)
    try:
        model.load_state_dict(torch.load(TCN_MODEL_PATH, map_location=DEVICE))
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    model.eval()
    
    # 2. Setup MediaPipe
    print("Initializing MediaPipe...")
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # 3. Setup Camera
    cap = cv2.VideoCapture(0)
    
    # Game State Variables
    state = "IDLE" 
    start_time = 0
    buffer = [] 
    
    final_result_text = ""
    result_confidence = 0.0
    
    print("Ready. Press 'SPACE' to start. 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        h_img, w_img, _ = frame.shape
        
        # --- MEDIA PIPE DETECTION ---
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        current_box = None 
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            x_list = [lm.x * w_img for lm in hand_landmarks.landmark]
            y_list = [lm.y * h_img for lm in hand_landmarks.landmark]
            
            x_min, x_max = min(x_list), max(x_list)
            y_min, y_max = min(y_list), max(y_list)
            
            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2
            width = x_max - x_min
            height = y_max - y_min
            
            # Square + Pad (Same logic as training)
            box_size = max(width, height) + PAD
            
            x1 = int(center_x - box_size / 2)
            y1 = int(center_y - box_size / 2)
            x2 = int(center_x + box_size / 2)
            y2 = int(center_y + box_size / 2)
            
            # Clamp
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(w_img, x2); y2 = min(h_img, y2)
            
            current_box = (x1, y1, x2, y2)

        # --- CAPTURE HELPER ---
        def record_frame():
            if current_box:
                x1, y1, x2, y2 = current_box
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                    tensor_img = transform(pil_img)
                    buffer.append(tensor_img)
            elif len(buffer) > 0:
                # If hand lost momentarily, duplicate last frame to keep timing correct
                buffer.append(buffer[-1])

        # --- STATE MACHINE ---
        ui_text = ""
        ui_color = (255, 255, 255)
        
        if current_box:
            cv2.rectangle(frame, (current_box[0], current_box[1]), (current_box[2], current_box[3]), (0, 255, 0), 2)

        if state == "IDLE":
            ui_text = "Press SPACE"
            buffer = []
            
        elif state == "COUNTDOWN":
            elapsed = time.time() - start_time
            if elapsed < 1.0: 
                ui_text = "1..."
            elif elapsed < 2.0: 
                ui_text = "2..."
            elif elapsed < 3.0: 
                ui_text = "3..."
                ui_color = (0, 255, 255) # Yellow warning
                # *** EARLY START ***
                # We start recording during "3..." to catch the beginning of the motion
                record_frame() 
            else: 
                state = "RECORDING"
                
        elif state == "RECORDING":
            ui_text = "GO!"
            ui_color = (0, 255, 0) # Green
            
            # Continue recording
            record_frame()
            
            # Progress Bar
            cv2.rectangle(frame, (0, h_img-10), (int(w_img * len(buffer)/SEQ_LEN), h_img), (0, 255, 0), -1)

            if len(buffer) >= SEQ_LEN:
                state = "PREDICT"
                
        elif state == "PREDICT":
            seq = torch.stack(buffer).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                logits = model(seq)
                probs = torch.softmax(logits, dim=1)
                conf, pred_idx = torch.max(probs, 1)
                
                user_move = CLASS_MAP[pred_idx.item()]
                my_move = WINNING_MOVE[user_move]
                result_confidence = conf.item()
                
                final_result_text = f"You: {user_move} | I Play: {my_move}"
                print(f"Result: {final_result_text} ({result_confidence:.0%})")
                state = "RESULT"
                start_time = time.time()

        elif state == "RESULT":
            ui_text = final_result_text
            ui_color = (0, 0, 255)
            if time.time() - start_time > 4.0: state = "IDLE"

        # Draw UI
        cv2.rectangle(frame, (0, 0), (w_img, 80), (0, 0, 0), -1)
        cv2.putText(frame, ui_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, ui_color, 3)
        if state == "RESULT":
            cv2.putText(frame, f"Conf: {result_confidence:.0%}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)
        
        cv2.imshow("RPS Battle", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        if key == ord(' ') and state == "IDLE":
            state = "COUNTDOWN"
            start_time = time.time()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_game()