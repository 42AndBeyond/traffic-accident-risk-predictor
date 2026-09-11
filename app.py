import streamlit as st
import numpy as np
import pandas as pd
import joblib
import json
import tempfile
from ultralytics import YOLO

st.set_page_config(page_title="Traffic Accident Risk Predictor", layout="centered")
st.title("🚦 Traffic Monitoring & Accident Risk Prediction")

@st.cache_resource
def load_models():
    yolo_model = YOLO('yolov8n.pt')
    risk_model = joblib.load('accident_risk_model.pkl')
    with open('defaults.json') as f:
        defaults = json.load(f)
    return yolo_model, risk_model, defaults

yolo_model, risk_model, defaults = load_models()

vehicle_classes = ['car', 'truck', 'bus', 'motorcycle']
density_map = {"Low": 0, "Medium": 1, "High": 2}

uploaded_video = st.file_uploader("Upload a traffic video", type=["mp4", "avi", "mov"])

if uploaded_video is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_video.read())
    video_path = tfile.name
    st.video(video_path)

    with st.spinner("Detecting vehicles... this may take a minute"):
        results = yolo_model(video_path, stream=True)
        traffic_counts = []
        for r in results:
            count = 0
            for box in r.boxes:
                class_name = yolo_model.names[int(box.cls[0])]
                if class_name in vehicle_classes:
                    count += 1
            traffic_counts.append(count)

    avg_count = np.mean(traffic_counts)
    density_label = "Low" if avg_count < 5 else "Medium" if avg_count < 10 else "High"

    st.subheader("Results")
    st.write(f"**Average vehicles per frame:** {avg_count:.2f}")
    st.write(f"**Traffic density:** {density_label}")

    input_row = defaults.copy()
    input_row['Traffic_Density'] = density_map[density_label]
    input_df = pd.DataFrame([input_row])[risk_model.feature_names_in_]

    prob = risk_model.predict_proba(input_df)[0][1]
    st.write(f"**Accident risk probability:** {prob:.2f}")
    if prob >= 0.4:
        st.error("⚠️ ACCIDENT RISK DETECTED")
    else:
        st.success("✅ LOW RISK")