import cv2
import torch
import torch.nn as nn
import numpy as np
import torchvision.models as models
import time
import os
import threading
import queue
import mediapipe as mp
from collections import deque
from torchvision import transforms
from PIL import Image

# --- CONFIGURATION ---
script_dir = os.path.dirname(os.path.abspath(__file__))
TCN_MODEL_PATH = os.path.join(script_dir, "rps_tcn_model.pth") 

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters (Must match training)
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

# --- WORKER THREAD FOR PREDICTION ---
def prediction_worker(model, input_queue, result_queue):
    while True:
        # Wait for a sequence of frames
        frames_tensor = input_queue.get()
        if frames_tensor is None: break # Exit signal
        
        # Run Inference
        with torch.no_grad():
            logits = model(frames_tensor)
            probs = torch.softmax(logits, dim=1)
            conf, pred_idx = torch.max(probs, 1)
            
            user_move = CLASS_MAP[pred_idx.item()]
            my_move = WINNING_MOVE[user_move]
            result_confidence = conf.item()
            
            # Send result back to Main Thread
            result_queue.put((user_move, my_move, result_confidence))

def run_game():
    # 1. Load Model
    print(f"Loading TCN Brain...")
    model = GestureTCN(num_classes=3).to(DEVICE)
    try:
        model.load_state_dict(torch.load(TCN_MODEL_PATH, map_location=DEVICE))
    except Exception as e:
        print(f"Error: {e}")
        return
    model.eval()
    
    # 2. Start Prediction Thread
    input_queue = queue.Queue()
    result_queue = queue.Queue()
    thread = threading.Thread(target=prediction_worker, args=(model, input_queue, result_queue))
    thread.daemon = True
    thread.start()

    # 3. Setup MediaPipe
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    cap = cv2.VideoCapture(0)
    
    # State
    state = "IDLE" 
    start_time = 0
    
    # Rolling Buffer: Always keeps the last 64 frames ready
    # This eliminates the need to "wait and record"
    rolling_buffer = deque(maxlen=SEQ_LEN)
    
    final_result_text = ""
    result_confidence = 0.0
    
    print("Ready. Press 'SPACE' to start. 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        h_img, w_img, _ = frame.shape
        
        # --- PROCESSING (Always running) ---
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        # Smart Cropping Logic
        current_box = None
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            x_list = [lm.x * w_img for lm in hand_landmarks.landmark]
            y_list = [lm.y * h_img for lm in hand_landmarks.landmark]
            
            cx = (min(x_list) + max(x_list)) / 2
            cy = (min(y_list) + max(y_list)) / 2
            box_size = max(max(x_list)-min(x_list), max(y_list)-min(y_list)) + PAD
            
            x1 = max(0, int(cx - box_size/2))
            y1 = max(0, int(cy - box_size/2))
            x2 = min(w_img, int(cx + box_size/2))
            y2 = min(h_img, int(cy + box_size/2))
            
            current_box = (x1, y1, x2, y2)

        # Add to buffer (Always happening in background)
        if current_box:
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                rolling_buffer.append(transform(pil_img))
        elif len(rolling_buffer) > 0:
            rolling_buffer.append(rolling_buffer[-1]) # Keep buffer full even if hand lost

        # --- GAME LOGIC ---
        ui_text = ""
        ui_color = (255, 255, 255)

        if state == "IDLE":
            ui_text = "Press SPACE"
            
        elif state == "COUNTDOWN":
            elapsed = time.time() - start_time
            if elapsed < 1.0: ui_text = "1..."
            elif elapsed < 2.0: ui_text = "2..."
            elif elapsed < 2.5: 
                ui_text = "3..." 
                ui_color = (0, 255, 255)
            else:
                # INSTANT TRIGGER
                # We don't record now. We grab what we already recorded!
                if len(rolling_buffer) == SEQ_LEN:
                    state = "PREDICTING"
                    
                    # Prepare tensor and send to thread
                    seq = torch.stack(list(rolling_buffer)).unsqueeze(0).to(DEVICE)
                    input_queue.put(seq)
                else:
                    state = "IDLE" # Buffer not full yet
                    print("Error: Buffer not full. Show hand earlier.")

        elif state == "PREDICTING":
            ui_text = "Reading Mind..."
            ui_color = (255, 0, 255)
            
            # Check if thread is done (Non-blocking check)
            try:
                user_move, my_move, conf = result_queue.get_nowait()
                final_result_text = f"You: {user_move} | I Play: {my_move}"
                result_confidence = conf
                
                state = "RESULT"
                start_time = time.time()
            except queue.Empty:
                pass # Keep showing "Reading Mind..." while UI updates

        elif state == "RESULT":
            ui_text = final_result_text
            ui_color = (0, 255, 0)
            if time.time() - start_time > 4.0: state = "IDLE"

        # Draw
        if current_box:
            cv2.rectangle(frame, (current_box[0], current_box[1]), (current_box[2], current_box[3]), (0, 255, 0), 2)
        
        cv2.rectangle(frame, (0, 0), (w_img, 80), (0, 0, 0), -1)
        cv2.putText(frame, ui_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, ui_color, 3)
        if state == "RESULT":
            cv2.putText(frame, f"Conf: {result_confidence:.0%}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)

        cv2.imshow("Magic RPS", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): 
            input_queue.put(None) # Kill thread
            break
        if key == ord(' ') and state == "IDLE":
            state = "COUNTDOWN"
            start_time = time.time()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_game()