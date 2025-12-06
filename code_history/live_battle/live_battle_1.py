import cv2
import torch
import torch.nn as nn
import numpy as np
import torchvision.models as models
from collections import deque
from inference import get_model
from torchvision import transforms
from PIL import Image

import os  # Add this if you haven't imported it yet

# --- CONFIGURATION ---
ROBOFLOW_API_KEY = "K1RsKDud3xFEWD4hdbeu"
MODEL_ID = "hand-detection-e3e9a/12"

# 1. Get the directory where THIS script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Join it with the model filename
# This forces Python to look inside 'virtual_machine' no matter where the terminal is
TCN_MODEL_PATH = os.path.join(script_dir, "rps_tcn_model.pth") 

print(f"Looking for model at: {TCN_MODEL_PATH}") # Debug print to be sure

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEQ_LEN = 64
FRAME_SIZE = 224 # Input size (224х224 as in ResNet)
CLASS_MAP = {0: 'Rock', 1: 'Paper', 2: 'Scissor'}
WINNING_MOVE = {'Rock': 'Paper', 'Paper': 'Scissor', 'Scissor': 'Rock'}

# --- MODEL DEFINITIONS (Must match training code exactly) ---
class ResNetFrameEncoder(nn.Module):
    def __init__(self, feature_dim=64):
        super().__init__()
        resnet = models.resnet18(weights=None) # No need to download weights again, we load ours
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
        # Hyperparams must match training
        self.encoder = ResNetFrameEncoder(feature_dim=64) 
        from pytorch_tcn import TCN
        self.tcn = TCN(
            num_inputs=64, num_channels=[64, 64], kernel_size=3,
            dropout=0.1, causal=True, input_shape='NCL' 
        )
        self.classifier = nn.Linear(64, num_classes)

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
    # 1. Load Object Detector
    print("Loading Roboflow Detector...")
    detector = get_model(model_id=MODEL_ID, api_key=ROBOFLOW_API_KEY)
    
    # 2. Load TCN
    print("Loading TCN Brain...")
    model = GestureTCN().to(DEVICE)
    model.load_state_dict(torch.load(TCN_MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    # 3. Start Webcam
    cap = cv2.VideoCapture(0)
    buffer = deque(maxlen=SEQ_LEN)
    
    # Smoothing buffer (averages last 3 predictions to stop flickering)
    prediction_buffer = deque(maxlen=3)
    
    print("Starting Battle! Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # A. Detect Hand
        results = detector.infer(frame)
        predictions = results[0].predictions if isinstance(results, list) else results.predictions
        
        # Find best box
        best_box = None
        max_conf = -1
        for pred in predictions:
            if pred.confidence > 0.4 and pred.confidence > max_conf:
                max_conf = pred.confidence
                best_box = pred

        # B. Crop and Process
        if best_box:
            # Coordinates
            x, y, w, h = best_box.x, best_box.y, best_box.width, best_box.height
            x1 = int(x - w/2) - 20
            y1 = int(y - h/2) - 20
            x2 = int(x + w/2) + 20
            y2 = int(y + h/2) + 20
            
            # Safe Crop
            h_img, w_img, _ = frame.shape
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_img, x2), min(h_img, y2)
            
            crop = frame[y1:y2, x1:x2]
            
            if crop.size > 0:
                # Draw Box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Prepare for Model
                pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                tensor_img = transform(pil_img)
                buffer.append(tensor_img)
        
        # C. Predict
        ui_text = "Watching..."
        ui_color = (255, 255, 255)
        
        if len(buffer) == SEQ_LEN:
            seq = torch.stack(list(buffer)).unsqueeze(0).to(DEVICE) # (1, 16, 3, 128, 128)
            
            with torch.no_grad():
                logits = model(seq)
                probs = torch.softmax(logits, dim=1)
                conf, pred_idx = torch.max(probs, 1)
                
                # Update smoothing buffer
                prediction_buffer.append((pred_idx.item(), conf.item()))
                
                # LOGIC: If high confidence, display counter move
                # We take the average confidence of the last few frames to be stable
                avg_conf = sum([p[1] for p in prediction_buffer]) / len(prediction_buffer)
                most_common_pred = max(set([p[0] for p in prediction_buffer]), key=[p[0] for p in prediction_buffer].count)
                
                if avg_conf > 0.70: # 70% Confidence Threshold
                    user_move = CLASS_MAP[most_common_pred]
                    my_move = WINNING_MOVE[user_move]
                    
                    ui_text = f"User: {user_move} | I PLAY: {my_move}"
                    ui_color = (0, 0, 255) # Red for challenge
                else:
                    ui_text = "Analysing..."

        # D. Draw UI
        cv2.putText(frame, ui_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, ui_color, 2)
        
        cv2.imshow("Rock Paper Scissors AI", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_game()