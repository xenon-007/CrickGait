# CrickGait 🏏 — Intelligent Cricket Shot and Footwork Analysis

CrickGait is a full-stack IoT + ML + Web project that analyzes a batsman's cricket shots and footwork in real-time using a combination of pose estimation and piezoelectric sensor data. Built for Raspberry Pi 5, it provides live video streaming, sensor graphing, shot classification, and future extension into autonomous buggy movement.

## 🔧 Components

### Hardware
- **Raspberry Pi 5**
- **ESP32** with Wi-Fi
- **Piezoelectric sensors** (placed in cricket shoes)
- **Mecanum wheel buggy** with motor driver (optional, for camera repositioning)

### Software Stack
| Layer | Tools/Tech |
|-------|------------|
| Backend | Flask, Python, scikit-learn, MediaPipe |
| Frontend | Streamlit |
| Hardware Control | GPIO, socket/web interface |
| ML Model | Random Forest / Logistic Regression (Pickle `.pkl`) |
| Data Transport | HTTP, JSON, timestamp alignment |

---

## 📦 Features

- 🔴 **Live Video Feed** from Raspberry Pi Camera
- 🦶 **Foot Pressure Sensor Data** Visualization via Streamlit
- 🧠 **ML Shot Classification** (Drive, Pull, Flick, Defensive)
- 🦾 (Optional) **Autonomous Buggy Repositioning**
- 🌐 **Accessible via LAN** on any device connected to the same network

