
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode

from webcam.streamlit_processor import process_frame

st.set_page_config(
    page_title="ConfidentAI Camera Test",
    layout="centered"
)

st.title("ConfidentAI — Live Emotion Detection")
st.write(
    "Allow camera access to test face detection and emotion prediction."
)

webrtc_streamer(
    key="emotion-test",
    mode=WebRtcMode.SENDRECV,
    video_frame_callback=process_frame,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    async_processing=True
)
