import streamlit as st
import numpy as np
import pandas as pd
import joblib
import json
import tempfile
import time
import av
import cv2
from ultralytics import YOLO
import streamlit.components.v1 as components
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

st.set_page_config(page_title="Traffic Accident Risk Predictor", page_icon="◈", layout="centered")

# ---------- CUSTOM STYLING ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #06080d;
    background-image:
        radial-gradient(circle at 15% 0%, rgba(56,189,248,0.06) 0%, transparent 45%),
        radial-gradient(circle at 85% 100%, rgba(56,189,248,0.04) 0%, transparent 40%);
}

.header-row {
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 6px;
}

.logo-icon {
    flex-shrink: 0;
}

.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 32px;
    font-weight: 700;
    color: #eef2f8;
    letter-spacing: -0.5px;
    margin: 0;
}

.hero-subtitle {
    color: #8b96ac;
    font-size: 14px;
    margin: 6px 0 34px 0;
    padding-left: 62px;
}

[data-testid="stFileUploader"], [data-testid="stCameraInput"] {
    background-color: #0c1017;
    border: 1.5px dashed #232b3a;
    border-radius: 14px;
    padding: 10px;
    transition: border-color 0.25s ease;
}

[data-testid="stFileUploader"]:hover, [data-testid="stCameraInput"]:hover {
    border-color: #38bdf8;
}

.section-label {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 2.5px;
    color: #6b7690;
    text-transform: uppercase;
    margin: 32px 0 10px 0;
    display: flex;
    align-items: center;
    gap: 10px;
}

.section-label::after {
    content: "";
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, #232b3a, transparent);
}

.result-banner {
    border-radius: 14px;
    padding: 18px 24px;
    margin: 20px 0 22px 0;
    display: flex;
    align-items: center;
    gap: 14px;
    animation: slideIn 0.45s cubic-bezier(0.16, 1, 0.3, 1);
}

.result-banner.high {
    background: linear-gradient(90deg, rgba(239,68,68,0.10), rgba(239,68,68,0.02));
    border: 1px solid rgba(239,68,68,0.35);
}

.result-banner.low {
    background: linear-gradient(90deg, rgba(52,211,153,0.10), rgba(52,211,153,0.02));
    border: 1px solid rgba(52,211,153,0.35);
}

.result-banner-text {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 17px;
    font-weight: 700;
    letter-spacing: 0.3px;
}

.pulse-dot {
    width: 9px;
    height: 9px;
    border-radius: 50%;
    flex-shrink: 0;
    animation: pulse 1.8s infinite;
}

.pulse-dot.high { background: #ef4444; }
.pulse-dot.low { background: #34d399; }

@keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(239,68,68,0.5); }
    70% { box-shadow: 0 0 0 9px rgba(239,68,68,0); }
    100% { box-shadow: 0 0 0 0 rgba(239,68,68,0); }
}

.metric-card {
    background: linear-gradient(165deg, #10151f, #0a0d14);
    border: 1px solid #1b2230;
    border-radius: 14px;
    padding: 20px 10px;
    text-align: center;
    animation: fadeUp 0.5s ease both;
    transition: transform 0.2s ease, border-color 0.2s ease;
}

.metric-card:hover {
    transform: translateY(-3px);
    border-color: #38bdf8;
}

.metric-label {
    font-size: 10.5px;
    letter-spacing: 1.5px;
    color: #6b7690;
    text-transform: uppercase;
    margin-bottom: 8px;
}

.metric-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 26px;
    font-weight: 700;
    color: #eef2f8;
}

@keyframes fadeUp {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes slideIn {
    from { opacity: 0; transform: translateX(-8px); }
    to { opacity: 1; transform: translateX(0); }
}

.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
}

.stTabs [data-baseweb="tab"] {
    background-color: #0c1017;
    border-radius: 10px 10px 0 0;
    color: #6b7690;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 13px;
    letter-spacing: 1px;
}

.stTabs [aria-selected="true"] {
    color: #38bdf8 !important;
    background-color: #10151f !important;
}

footer {visibility: hidden;}
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown("""
<div class="header-row">
    <svg class="logo-icon" width="42" height="42" viewBox="0 0 48 48" fill="none">
        <circle cx="24" cy="24" r="21" stroke="#38bdf8" stroke-width="1.5" opacity="0.35"/>
        <circle cx="24" cy="24" r="14" stroke="#38bdf8" stroke-width="1.5" opacity="0.6"/>
        <circle cx="24" cy="24" r="3.5" fill="#38bdf8"/>
        <line x1="24" y1="3" x2="24" y2="9" stroke="#38bdf8" stroke-width="1.5"/>
        <line x1="24" y1="39" x2="24" y2="45" stroke="#38bdf8" stroke-width="1.5"/>
        <line x1="3" y1="24" x2="9" y2="24" stroke="#38bdf8" stroke-width="1.5"/>
        <line x1="39" y1="24" x2="45" y2="24" stroke="#38bdf8" stroke-width="1.5"/>
    </svg>
    <h1 class="hero-title">Traffic Monitoring &amp; Accident Risk Prediction</h1>
</div>
<div class="hero-subtitle">Vehicle detection combined with a trained accident-risk model — analyze a video or a live camera feed.</div>
""", unsafe_allow_html=True)

# ---------- VOICE UNLOCK ----------
if st.button("Enable Voice Alerts"):
    components.html("""
        <script>
        var unlock = new SpeechSynthesisUtterance(" ");
        window.speechSynthesis.speak(unlock);
        </script>
    """, height=0)
    st.success("Voice alerts enabled for this session")

# ---------- MODEL LOADING ----------
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

# ---------- VOICE ALERT (browser text-to-speech) ----------
def speak(text, rate=0.95, pitch=1.0):
    safe_text = text.replace('"', '').replace("'", "")
    components.html(f"""
        <script>
        function speakNow() {{
            if (!window.speechSynthesis) return;
            window.speechSynthesis.cancel();
            var msg = new SpeechSynthesisUtterance("{safe_text}");
            msg.rate = {rate};
            msg.pitch = {pitch};
            window.speechSynthesis.speak(msg);
        }}
        var voices = window.speechSynthesis.getVoices();
        if (voices.length > 0) {{
            speakNow();
        }} else {{
            window.speechSynthesis.onvoiceschanged = speakNow;
        }}
        </script>
    """, height=0)

def run_risk_model(density_label):
    input_row = defaults.copy()
    input_row['Traffic_Density'] = density_map[density_label]
    input_df = pd.DataFrame([input_row])[risk_model.feature_names_in_]
    prob = risk_model.predict_proba(input_df)[0][1]
    return prob

def show_results(avg_count, density_label, prob, voice_enabled=True, session_key="last_risk_level"):
    st.markdown('<div class="section-label">Analysis Results</div>', unsafe_allow_html=True)

    current_level = "high" if prob >= 0.4 else "low"
    previous_level = st.session_state.get(session_key, None)
    should_announce = voice_enabled and current_level == "high" and current_level != previous_level
    st.session_state[session_key] = current_level

    if current_level == "high":
        st.markdown("""
        <div class="result-banner high">
            <div class="pulse-dot high"></div>
            <span class="result-banner-text" style="color:#f87171;">HIGH ACCIDENT RISK</span>
        </div>
        """, unsafe_allow_html=True)
        if should_announce:
            speak(f"High accident risk. Traffic density {density_label}.")
    else:
        st.markdown("""
        <div class="result-banner low">
            <div class="pulse-dot low"></div>
            <span class="result-banner-text" style="color:#34d399;">LOW ACCIDENT RISK</span>
        </div>
        """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Avg Vehicles</div>
            <div class="metric-value">{avg_count:.1f}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Traffic Density</div>
            <div class="metric-value">{density_label}</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Risk Probability</div>
            <div class="metric-value">{prob:.0%}</div>
        </div>""", unsafe_allow_html=True)

# ---------- LIVE CAMERA PROCESSOR ----------
class RiskProcessor(VideoProcessorBase):
    def __init__(self):
        self.last_check = 0
        self.interval = 4
        self.latest_count = 0
        self.latest_density = "Low"
        self.latest_prob = 0.0
        self.result_ready = False

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        now = time.time()

        if now - self.last_check >= self.interval:
            self.last_check = now
            results = yolo_model(img, verbose=False)
            count = 0
            for box in results[0].boxes:
                class_name = yolo_model.names[int(box.cls[0])]
                if class_name in vehicle_classes:
                    count += 1

            density_label = "Low" if count < 5 else "Medium" if count < 10 else "High"
            prob = run_risk_model(density_label)

            self.latest_count = count
            self.latest_density = density_label
            self.latest_prob = prob
            self.result_ready = True

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ---------- TABS ----------
tab1, tab2 = st.tabs(["UPLOAD VIDEO", "LIVE CAMERA"])

# ---------- TAB 1: VIDEO UPLOAD (with frame skipping for speed) ----------
with tab1:
    voice_on_tab1 = st.toggle("Voice alerts", value=True, key="voice_tab1")

    st.markdown('<div class="section-label">Upload Video</div>', unsafe_allow_html=True)
    uploaded_video = st.file_uploader("Upload a traffic video", type=["mp4", "avi", "mov"], label_visibility="collapsed")

    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())
        video_path = tfile.name
        st.video(video_path)

        with st.spinner("Detecting vehicles..."):
            cap = cv2.VideoCapture(video_path)
            frame_skip = 10
            traffic_counts = []
            frame_idx = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % frame_skip == 0:
                    results = yolo_model(frame, verbose=False)
                    count = 0
                    for box in results[0].boxes:
                        class_name = yolo_model.names[int(box.cls[0])]
                        if class_name in vehicle_classes:
                            count += 1
                    traffic_counts.append(count)
                frame_idx += 1

            cap.release()

        avg_count = np.mean(traffic_counts) if traffic_counts else 0
        density_label = "Low" if avg_count < 5 else "Medium" if avg_count < 10 else "High"
        prob = run_risk_model(density_label)

        show_results(avg_count, density_label, prob, voice_enabled=voice_on_tab1, session_key="last_risk_video")

# ---------- TAB 2: LIVE CAMERA (AUTO-CAPTURE, with STUN server fix) ----------
with tab2:
    voice_on_tab2 = st.toggle("Voice alerts", value=True, key="voice_tab2")

    st.markdown('<div class="section-label">Live Camera</div>', unsafe_allow_html=True)
    st.caption("Camera streams continuously — analysis updates automatically every few seconds.")

    ctx = webrtc_streamer(
        key="live-traffic",
        video_processor_factory=RiskProcessor,
        media_stream_constraints={"video": True, "audio": False},
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
    )

    result_placeholder = st.empty()

    if ctx.video_processor:
        while ctx.state.playing:
            if ctx.video_processor.result_ready:
                with result_placeholder.container():
                    show_results(
                        ctx.video_processor.latest_count,
                        ctx.video_processor.latest_density,
                        ctx.video_processor.latest_prob,
                        voice_enabled=voice_on_tab2,
                        session_key="last_risk_camera"
                    )
            time.sleep(1)