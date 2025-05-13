import os
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle

# Initialize MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False)
pose_landmarks = 17  # Use 17 major joints

def extract_pose_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    keypoints = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize and preprocess
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        if results.pose_landmarks:
            frame_keypoints = []
            for i, lm in enumerate(results.pose_landmarks.landmark):
                if i >= pose_landmarks:
                    break
                frame_keypoints.extend([lm.x, lm.y, lm.z])
            keypoints.append(frame_keypoints)

    cap.release()

    if not keypoints:
        return None

    # Average over all frames
    return np.mean(np.array(keypoints), axis=0)

# Load videos and extract features
def build_dataset(root_dir):
    X, y = [], []
    for label in os.listdir(root_dir):
        label_dir = os.path.join(root_dir, label)
        for file in os.listdir(label_dir):
            video_path = os.path.join(label_dir, file)
            print(f"Processing: {video_path}")
            features = extract_pose_from_video(video_path)
            if features is not None:
                X.append(features)
                y.append(label)
    return np.array(X), np.array(y)

# Load and process dataset
X, y = build_dataset("cricket_shots")

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Save the model
with open("cricket_shot_model.pkl", "wb") as f:
    pickle.dump(rf, f)

# Save the label encoder
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print("Model training complete and saved as 'cricket_shot_model.pkl'.")
