# ------------------- BACKEND (Flask + MQTT on Raspberry Pi) -------------------
import threading
import time
import paho.mqtt.client as mqtt
from flask import Flask, Response, jsonify
from datetime import datetime
import cv2
import mediapipe as mp
import numpy as np
import pickle

app = Flask(_name_)

# Globals
latest_data = {"sensor1": 0.0, "sensor2": 0.0, "timestamp": ""}
predicted_shot = "Stance"

# Load ML model
try:
    with open("cricket_shot_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
except Exception as e:
    model, label_encoder = None, None

# MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# MQTT Config
broker = "broker.hivemq.com"
port = 1883
topic = "test/esp32"

def update_sensor_data(sensor1, sensor2):
    global latest_data
    latest_data = {
        "sensor1": sensor1,
        "sensor2": sensor2,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

def on_message(client, userdata, msg):
    try:
        payload = msg.payload.decode()
        parts = payload.split(",")
        sensor1_val = float(parts[0].split(":")[1])
        sensor2_val = float(parts[1].split(":")[1])
        update_sensor_data(sensor1_val, sensor2_val)
    except:
        pass

def mqtt_thread():
    client = mqtt.Client()
    client.on_message = on_message
    client.connect(broker, port, 60)
    client.subscribe(topic)
    client.loop_forever()

threading.Thread(target=mqtt_thread, daemon=True).start()

# Camera & Pose
cap = cv2.VideoCapture(0)

def extract_pose_features(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)
    if results.pose_landmarks:
        keypoints = []
        for lm in results.pose_landmarks.landmark[:17]:
            keypoints.extend([lm.x, lm.y, lm.z])
        return np.array(keypoints).reshape(1, -1)
    return None

def generate_frames():
    global predicted_shot
    while True:
        success, frame = cap.read()
        if not success:
            continue
        features = extract_pose_features(frame)
        if features is not None and model and label_encoder:
            pred = model.predict(features)
            predicted_shot = label_encoder.inverse_transform(pred)[0]
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/live_sensor_data')
def live_sensor_data():
    return jsonify(latest_data)

@app.route('/shot_label')
def shot_label():
    return jsonify({"label": predicted_shot})

if _name_ == '_main_':
    app.run(host="0.0.0.0", port=5000)

