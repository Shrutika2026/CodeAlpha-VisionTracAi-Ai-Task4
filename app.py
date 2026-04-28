import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import cv2
import av
from ultralytics import YOLO

# --- PAGE CONFIG ---
st.set_page_config(page_title="VisionTrac Ai", layout="wide")

# --- CUSTOM UI & BACKGROUND ---
st.markdown("""
    <style>
    /* Import Google Fonts - Bree Serif added */
    @import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&family=Bree+Serif&display=swap');

    /* Background and Layout */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: white;
    }
    
    /* Left-Aligned Title */
    .main-title {
        font-family: 'Press Start 2P', cursive !important;
        font-size: 28px !important; 
        color: #00d2ff !important;
        text-shadow: 2px 2px 5px rgba(0, 0, 0, 0.7);
        text-align: left !important;
        margin-left: 50px;
        margin-top: 20px;
        display: block;
    }
    
    .img-container {
        text-align: left !important;
        margin-left: 50px;
        margin-top: 20px;
    }

    .bottom-container {
        text-align: left !important;
        margin-left: 50px;
        margin-top: 20px;
        width: 100%;
    }
    
    div.stButton > button:first-child {
        background-color: #00d2ff;
        color: #0f0c29;
        font-weight: bold;
        padding: 15px 30px;
        border-radius: 10px;
        border: none;
    }
    </style>
    """, unsafe_allow_html=True)

# --- SESSION STATE FOR NAVIGATION ---
if 'entered' not in st.session_state:
    st.session_state.entered = False

# --- LANDING PAGE ---
if not st.session_state.entered:
    st.markdown('<h1 class="main-title">VisionTrac AI</h1>', unsafe_allow_html=True)
    st.markdown('<div class="img-container">', unsafe_allow_html=True)
    st.image("https://thumbs.dreamstime.com/b/cartoon-business-man-standing-holding-big-magnifying-glass-cartoon-business-man-standing-holding-big-magnifying-glass-324552864.jpg", 
             width=400)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="bottom-container">', unsafe_allow_html=True)
    st.markdown('<div style="font-size: 20px; color: white; margin-bottom: 20px;">Precision Detection at Your Fingertips</div>', unsafe_allow_html=True)
    if st.button("Launch VisionTrac Engine"):
        st.session_state.entered = True
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# --- ACTUAL DETECTION PAGE ---
else:
    col1, col2 = st.columns([1, 10])
    with col1:
        if st.button("← Back"):
            st.session_state.entered = False
            st.rerun()

    # Model is loaded once to prevent reloading lag
    if "model" not in st.session_state:
        st.session_state.model = YOLO('yolov8n')

    st.markdown('<p style="font-family: \'Bree Serif\', serif; font-size: 50px; color: #00d2ff; font-weight: bold;">VisionTrac Ai: Real-Time Detection</p>', unsafe_allow_html=True)
    st.write("Detecting: People, Scissors, Bottles, Laptops, and 76 more!")

    def video_frame_callback(frame):
        img = frame.to_ndarray(format="bgr24")
        
        # We keep conf=0.4 (original sensitivity) so it detects many objects
        # We use stream=True and a small imgsz for speed
        results = st.session_state.model.predict(img, conf=0.4, imgsz=224, stream=True, verbose=False)
        
        annotated_frame = img
        for r in results:
            annotated_frame = r.plot()
            break 
        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    webrtc_streamer(
        key="live-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_frame_callback=video_frame_callback,
        media_stream_constraints={
            "video": {
                "width": {"ideal": 640}, 
                "height": {"ideal": 480},
                "frameRate": {"ideal": 20}, # Smoother frame rate
                "facingMode": "environment",
            },
            "audio": False
        },
        async_processing=True, 
    )
