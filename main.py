#from model.model import InceptionI3d
import streamlit as st
import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
import json
from collections import deque
import torch.nn as nn
from torchvision.models.video import mvit_v2_s
# Load glosses and initialize model
def load_glosses(json_path):
    with open(json_path, 'r') as f:
        gloss_dict = json.load(f)
    return gloss_dict

glosses = load_glosses('./assets/glosses.json')
num_classes = len(glosses)
checkpoint = 'assets/best_model_mvit_run2.pth'

# model = InceptionI3d(400, in_channels=3)
# model.load_state_dict(torch.load('assets/rgb_imagenet.pt'))
# model.replace_logits(num_classes)

model = mvit_v2_s(weights=None)
model.head = nn.Sequential(
    nn.Dropout(p=0.5, inplace=True),
    nn.Linear(model.head[1].in_features, num_classes)
)

model.load_state_dict(torch.load(checkpoint,  map_location='cuda:0'))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

st.title("Real-Time Sign Language Recognition App")
st.markdown("This app captures real-time video input and translates sign language gestures to text. Press **Start/Stop Video Capture** to begin or end.")

if 'run_app' not in st.session_state:
    st.session_state.run_app = False

start_stop_button = st.button('Start/Stop Video Capture')
if start_stop_button:
    st.session_state.run_app = not st.session_state.run_app

# Streamlit placeholders
frame_window = st.image([])
output_text = st.empty()

# Confidence threshold for displaying predictions
CONFIDENCE_THRESHOLD = 0.5

# Function to crop the frame
def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 3) - (min_dim // 5)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y : start_y + min_dim, start_x : start_x + min_dim]


def compress_frames(frames, target_frames=16):
    compressed = frames[::2]

    # If the result exceeds the target, evenly space the compressed frames
    if len(compressed) > target_frames:
        frame_indices = np.linspace(0, len(compressed) - 1, target_frames, dtype=int)
        compressed = [compressed[i] for i in frame_indices]

    return torch.stack(compressed)


# Real-time frame processing and continuous prediction
if st.session_state.run_app:
    cap = cv2.VideoCapture(0)
    frames = deque(maxlen=40)  # Sliding window of 60 frames
    outputs = []
    prev_frame = None

    while st.session_state.run_app:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture video")
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_window.image(frame, channels='RGB')
        frame = crop_center_square(frame)
        image = Image.fromarray(frame)
        transformed_frame = transform(image)
        frames.append(transformed_frame)

        if len(frames) == 40:
            compressed_frames = compress_frames(list(frames), target_frames=16)
            # Add batch dimension and permute for model input
            input_tensor = compressed_frames.unsqueeze(0).permute(0, 2, 1, 3, 4).to(device)
            # Stack frames and add batch dimension
            # Model inference
            with torch.no_grad():
                output = model(input_tensor)
            
            labels = torch.argmax(output, dim=1).item()
            # Display only if confidence exceeds the threshold
            gesture_gloss = glosses.get(str(labels), " ")
            outputs.append(gesture_gloss)
            output_text.markdown(f"**Recognized Gesture:** {' '.join(outputs)}")
            frames.clear()

    cap.release()
