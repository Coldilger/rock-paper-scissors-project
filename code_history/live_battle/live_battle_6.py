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
script_dir = os.path.dirname(os.path.abspath(__file__))
TCN_MODEL_PATH = os.path.join(script_dir, "rps_tcn_model.pth") 

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters (Must match training!)
SEQ_LEN = 64
FRAME_SIZE = 224
FEATURE_DIM = 128
CHANNELS = [64, 64, 64, 64]
KERNEL_SIZE = 6
PAD = 50 

CLASS_MAP = {0: 'Rock', 1: 'Paper', 2: 'Scissor'}
WINNING_MOVE = {'Rock': 'Paper', 'Paper': 'Scissor', 'Scissor': 'Rock'}

# --- SPLIT MODEL ARCHITECTURE ---
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

class GestureTCNHead(nn.Module):
    def __init__(self, feature_dim=128, num_classes=3):
        super().__init__()
        from pytorch_tcn import TCN
        self.tcn = TCN(
            num_inputs=feature_dim, 
            num_channels=CHANNELS, 
            kernel_size=KERNEL_SIZE,
            dropout=0.2, 
            causal=True, 
            input_shape='NCL' 
        )
        self.classifier = nn.Linear(CHANNELS[-1], num_classes)

    def forward(self, features):
        features = features.permute(0, 2, 1)
        tcn_out = self.tcn(features)
        last_out = tcn_out[:, :, -1]
        return self.classifier(last_out)

class UnifiedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ResNetFrameEncoder(feature_dim=FEATURE_DIM)
        self.decoder = GestureTCNHead(feature_dim=FEATURE_DIM, num_classes=3)

# --- UTILS ---
transform = transforms.Compose([
    transforms.Resize((FRAME_SIZE, FRAME_SIZE)),
    transforms.ToTensor(),
])

def run_game():
    # 1. Load Model
    print(f"Loading AI Brain...")
    full_model = UnifiedModel().to(DEVICE)
    try:
        checkpoint = torch.load(TCN_MODEL_PATH, map_location=DEVICE)
        new_state_dict = {}
        for key, value in checkpoint.items():
            if key.startswith("tcn.") or key.startswith("classifier."):
                new_key = f"decoder.{key}"
            else:
                new_key = key
            new_state_dict[new_key] = value
        full_model.load_state_dict(new_state_dict)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    full_model.eval()
    encoder = full_model.encoder.to(DEVICE)
    decoder = full_model.decoder.to(DEVICE)
    
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
    
    # --- BUFFERS ---
    feature_buffer = deque(maxlen=SEQ_LEN)
    last_features = None 
    
    state = "IDLE" 
    start_time = 0
    final_result_text = ""
    result_confidence = 0.0
    
    print("Ready. Press 'SPACE' to start. 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        h_img, w_img, _ = frame.shape
        
        # --- 1. CONTINUOUS PROCESSING (Zero Lag) ---
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        current_features = None
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            x_list = [lm.x * w_img for lm in hand_landmarks.landmark]
            y_list = [lm.y * h_img for lm in hand_landmarks.landmark]
            
            cx = (min(x_list) + max(x_list)) / 2
            cy = (min(y_list) + max(y_list)) / 2
            box_size = max(max(x_list)-min(x_list), max(y_list)-min(y_list)) + PAD
            
            x1 = max(0, int(cx - box_size/2))
            y1 = max(0, int(cy - box_size/2))
            x2 = min(w_img, int(cx + box_size/2))
            y2 = min(h_img, int(cy + box_size/2))
            
            if x2 > x1 and y2 > y1:
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                    tensor_img = transform(pil_img).unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        current_features = encoder(tensor_img).cpu().numpy().flatten()
                        last_features = current_features
        elif last_features is not None:
            current_features = last_features

        if current_features is not None:
            feature_buffer.append(current_features)
        
        # --- 2. GAME LOGIC (The Magic Timing) ---
        ui_text = ""
        ui_color = (255, 255, 255)

        if state == "IDLE":
            ui_text = "Press SPACE"
            
        elif state == "COUNTDOWN":
            elapsed = time.time() - start_time
            
            if elapsed < 1.5: 
                ui_text = "1..."
            elif elapsed < 3.0: 
                ui_text = "2..."
            elif elapsed < 4.2: 
                ui_text = "3..." 
                ui_color = (0, 255, 255) # Yellow -> Get ready!
            else:
                # INSTANT TRIGGER at 4.2s
                # We skip the "Recording" text. We just predict.
                state = "PREDICT"
                
        elif state == "PREDICT":
            # Safety fill if buffer is slightly empty
            while len(feature_buffer) < SEQ_LEN and len(feature_buffer) > 0:
                feature_buffer.append(feature_buffer[-1])

            if len(feature_buffer) == SEQ_LEN:
                seq_features = torch.tensor(np.array(feature_buffer), dtype=torch.float32).unsqueeze(0).to(DEVICE)
                
                with torch.no_grad():
                    logits = decoder(seq_features)
                    probs = torch.softmax(logits, dim=1)
                    conf, pred_idx = torch.max(probs, 1)
                    
                    user_move = CLASS_MAP[pred_idx.item()]
                    my_move = WINNING_MOVE[user_move]
                    result_confidence = conf.item()
                    
                    final_result_text = f"You: {user_move} | I Play: {my_move}"
                    state = "RESULT"
                    start_time = time.time()
            else:
                ui_text = "Hold on..." # Rare case: no hand detected at all

        elif state == "RESULT":
            ui_text = final_result_text
            ui_color = (0, 255, 0) # Green for Victory
            
            if time.time() - start_time > 4.0: 
                state = "IDLE"
                feature_buffer.clear()

        # --- DRAW UI ---
        cv2.rectangle(frame, (0, 0), (w_img, 80), (0, 0, 0), -1)
        cv2.putText(frame, ui_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, ui_color, 3)
        if state == "RESULT":
            cv2.putText(frame, f"Conf: {result_confidence:.0%}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)

        cv2.imshow("Magic RPS", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        if key == ord(' ') and state == "IDLE":
            state = "COUNTDOWN"
            feature_buffer.clear()
            start_time = time.time()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_game()