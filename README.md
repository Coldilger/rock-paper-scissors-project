# rock-paper-scissors-project
**Deadline: 2nd of December**
# Planned stages of the project
1) Hand detection
2) Identification of one on the 4 classes (r, p, s and other)
3) Live detection of the specific gestures
4) Improving Time To Detect (TTD) by fine-tuning on slight movements which give away the intended symbol to be played

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

# datasets:
https://universe.roboflow.com/roboflow-58fyf/rock-paper-scissors-sxsw/dataset/14


# project ost:
https://open.spotify.com/track/07UfHum6GRuQSSp3QwxGM7?si=c2988149e8f54bb4
