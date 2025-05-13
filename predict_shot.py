import cv2
import mediapipe as mp
import numpy as np
import pickle
import sys

# Load model and label encoder
with open("cricket_shot_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False)

def extract_pose_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    keypoints = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize and convert to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        if results.pose_landmarks:
            frame_keypoints = []
            for i, lm in enumerate(results.pose_landmarks.landmark[:17]):  # First 17 keypoints
                frame_keypoints.extend([lm.x, lm.y, lm.z])
            keypoints.append(frame_keypoints)

    cap.release()

    if not keypoints:
        return None

    # Average across frames
    return np.mean(np.array(keypoints), axis=0).reshape(1, -1)

def predict_shot(video_path):
    print(f"Processing video: {video_path}")
    features = extract_pose_from_video(video_path)

    if features is None:
        print("No pose detected in the video.")
        return

    pred = model.predict(features)
    label = label_encoder.inverse_transform(pred)[0]
    print(f"Predicted Shot Type: {label}")

# Change the path below to your test video
if __name__ == "__main__":
    # Instead of using sys.argv
    video_path = r"C:\Users\hp\Desktop\RPW2\test_videos\shot6.mp4"
    predict_shot(video_path)
