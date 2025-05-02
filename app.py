import streamlit as st
import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image
import joblib

# Load the MobileNetV2 feature extractor
mobilenet = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
mobilenet.classifier = torch.nn.Identity()  # Remove classification layer
mobilenet.eval()

# Load the trained SVM model
svm_model = joblib.load('svm_sign_language_model.pkl')

# Load the Label Encoder
label_encoder = joblib.load('label_encoder.pkl')

# Preprocessing

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Feature Extraction Function

@torch.no_grad()
def extract_features_from_video(video_path, max_frames=64):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % 2 == 0:  
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = transform(frame)
            frames.append(frame)
        frame_count += 1
        if len(frames) >= max_frames:
            break
    cap.release()

    if not frames:
        return None

    frames_tensor = torch.stack(frames) 
    features = mobilenet(frames_tensor)
    avg_feature = features.mean(dim=0)
    return avg_feature.numpy()

# Streamlit App

st.title("Norwegian Sign Language (NSL) Recognition")

st.write("""
Upload a short video of a healthcare-related sign in Norwegian Sign Language (NSL).
The system will analyze the video and predict the corresponding healthcare term.
""")

uploaded_file = st.file_uploader("Upload a video file", type=["mp4"])

if uploaded_file is not None:
    st.video(uploaded_file)

    # Save uploaded video to disk temporarily
    with open("temp_video.mp4", "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.info("Extracting features and predicting... Please wait.")

    # Extract features
    avg_feature = extract_features_from_video("temp_video.mp4")

    if avg_feature is not None:
        # Predict
        predicted_label_encoded = svm_model.predict(avg_feature.reshape(1, -1))
        predicted_label = label_encoder.inverse_transform(predicted_label_encoded)

        st.success(f"Predicted Sign: {predicted_label[0]}")
    else:
        st.error("Error: No frames extracted from video. Please upload a valid video.")