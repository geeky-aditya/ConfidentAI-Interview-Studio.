
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode

st.set_page_config(page_title="Camera Test", layout="centered")

st.title("ConfidentAI — Camera Test")
st.write("Allow camera access when your browser asks.")

webrtc_streamer(
    key="camera-test",
    mode=WebRtcMode.SENDRECV,
    media_stream_constraints={
        "video": True,
        "audio": False,
    },
    async_processing=True,
)

st.caption("If the video appears above, the browser camera connection works.")
