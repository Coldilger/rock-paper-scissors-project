# ZeroLag RPS
**Deadline: 2nd of December**
# Planned stages of the project
Real-time gesture recognition systems often suffer from la-
tency due to the classification of completed motions. This
report presents ZeroLag RPS, an AI agent designed to
defeat a human opponent in Rock-Paper-Scissors by pre-
dicting moves before gesture completion. We developed
a hybrid architecture combining a ResNet-18 spatial en-
coder with a Temporal Convolutional Network (TCN) head,
trained on a custom dataset of gesture sequences. The
TCN’s dilated causal convolutions efficiently model tempo-
ral dependencies within a 64-frame window, enabling the
detection of subtle “wind-up” micromovements.
The pipeline integrates MediaPipe for skeletal tracking
and utilizes a sigmoid time-weighted loss function to ad-
dress early-phase ambiguity. Our model achieves a test set
accuracy of 70.6% and demonstrates the ability to infer the
user’s move as early as frame 48 (approx. 200ms before
completion). Deployed in a live environment with a rolling
buffer, the system delivers zero-latency counter-moves for a
seamless user experience.

# Files
The git contains the following structure:
├── code/
│   ├── Videos/                 -> place here videos that need to be processed to obtain the dataset of frames
│   └── Video2frames.ipynb      -> helper code that creates frames from the videos
├── data/
│   └── final_split_dataset/    -> here we obtain the dataset
│       ├── test/
│       ├── train/
│       └── val/
├── noaudio_ver5/               -> this should be the dataset
├── data_splitting_ver3.py      -> final processing to go obtain the final split dataset we need from noaudio dataset
├── live_battle_10f.py          -> to play the live version of the game
├── README.md
├── rps_tcn_model.pth           -> model weights
└── TCN_test_Ilia_7.ipynb       -> main file with the TCN model

# Example 


https://github.com/user-attachments/assets/ddd86337-263c-498b-b0c7-1373e4ca9a76


# datasets:
https://drive.google.com/drive/folders/1yZMYdbUkMhnvQazs3OVpYUC7R1Ky9l4b?usp=sharing


