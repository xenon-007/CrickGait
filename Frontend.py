
# ------------------- FRONTEND (Streamlit Dashboard) -------------------


import streamlit as st
import requests
import pandas as pd
from streamlit_autorefresh import st_autorefresh

RASPI_IP = "x.x.x.x"  # Replace with actual Raspberry Pi IP

st.set_page_config(layout="wide")
st.title("CrickGait: Live Feed & Piezo Data")

st_autorefresh(interval=2000, key="refresh")

col1, col2 = st.columns([2, 1])

# --- Camera Feed ---
with col1:
    st.subheader("Live Camera Feed")
    st.markdown(
        f'<img src="http://{RASPI_IP}:5000/video_feed" width="640" height="480">',
        unsafe_allow_html=True
    )

# --- Shot Label ---
with col2:
    st.subheader("Shot Classification")
    try:
        r = requests.get(f"http://{RASPI_IP}:5000/shot_label")
        if r.status_code == 200:
            st.success(f"Shot: {r.json()['label']}")
        else:
            st.warning("Failed to fetch shot label")
    except:
        st.error("Server not reachable")

# --- Sensor Data ---
st.subheader("Piezo Sensor Data (Live)")

if 'sensor_df' not in st.session_state:
    st.session_state.sensor_df = pd.DataFrame(columns=["Timestamp", "Sensor 1", "Sensor 2"])

try:
    r = requests.get(f"http://{RASPI_IP}:5000/live_sensor_data")
    if r.status_code == 200:
        data = r.json()
        new_row = {
            "Timestamp": data["timestamp"],
            "Sensor 1": data["sensor1"],
            "Sensor 2": data["sensor2"]
        }
        st.session_state.sensor_df = pd.concat(
            [st.session_state.sensor_df, pd.DataFrame([new_row])],
            ignore_index=True
        )
except:
    st.warning("Sensor data not available")

if not st.session_state.sensor_df.empty:
    df = st.session_state.sensor_df.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df.set_index("Timestamp", inplace=True)
    st.line_chart(df[["Sensor 1", "Sensor 2"]])