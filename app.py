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
    @import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&display=swap');

    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: white;
        overflow: hidden;
    }
    
    .main-title {
        font-family: 'Press Start 2P', cursive;
        font-size: 80px; 
        color: #00d2ff;
        text-shadow: 4px 4px 10px rgba(0, 0, 0, 0.7);
        position: absolute;
        left: 0;
        right: 0;
        top: 8%;
        text-align: center;
    }
    
    /* Centering logic for Image and Button */
    .img-center, .button-center {
        position: absolute;
        left: 0;
        right: 0;
        width: 100%;
        display: flex;
        justify-content: center;
        align-items: center;
    }

    .img-center { top: 25%; }
    .button-center { top: 82%; }

    .subtext-center {
        position: absolute;
        left: 0;
        right: 0;
        top: 75%;
        text-align: center;
        font-size: 22px;
    }

    /* Force Streamlit widgets to center their content */
    [data-testid="stHorizontalBlock"], .stButton, .stImage {
        display: flex;
        justify-content: center;
        width: 100%;
    }

    div.stButton > button:first-child {
        background-color: #00d2ff;
        color: #0f0c29;
        font-weight: bold;
        padding: 18px 35px;
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
    st.markdown('<p class="main-title">VisionTrac AI</p>', unsafe_allow_html=True)
    
    # Image wrapper
    st.markdown('<div class="img-center">', unsafe_allow_html=True)
    st.image("https://thumbs.dreamstime.com/b/cartoon-business-man-standing-holding-big-magnifying-glass-cartoon-business-man-standing-holding-big-magnifying-glass-324552864.jpg", 
             width=420)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="subtext-center">Precision Detection at Your Fingertips</div>', unsafe_allow_html=True)
    
    # Button wrapper
    st.markdown('<div class="button-center">', unsafe_allow_html=True)
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

    model = YOLO('yolov8n')

    st.markdown('<p style="font-size: 40px; color: #00d2ff; font-weight: bold;">VisionTrac Ai: Real-Time Detection</p>', unsafe_allow_html=True)
    st.write("Detecting: People, Scissors, Bottles, Laptops, and 76 more!")

    def video_frame_callback(frame):
        img = frame.to_ndarray(format="bgr24")
        results = model.predict(img, conf=0.4, imgsz=256, stream=True)
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
                "width": 640, "height": 480, "frameRate": 15 
            },
            "audio": False
        },
        async_processing=True, 
    )