import time
import streamlit as st
import streamlit.components.v1 as components
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from webcam.streamlit_processor import (
    process_frame,
    session_metrics,
    pose_history
)

from webcam.confidence_score import calculate_confidence_score
from utils.helpers import generate_feedback



st.set_page_config(
    page_title="ConfidentAI | Interview Studio",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -------------------- SESSION STATE --------------------

if "interview_started" not in st.session_state:
    st.session_state.interview_started = False

if "interview_finished" not in st.session_state:
    st.session_state.interview_finished = False

if "selected_question" not in st.session_state:
    st.session_state.selected_question = (
        "Explain the four pillars of OOP."
    )

if "selected_duration" not in st.session_state:
    st.session_state.selected_duration = 60

if "interview_start_time" not in st.session_state:
    st.session_state.interview_start_time = None

if "time_expired" not in st.session_state:
    st.session_state.time_expired = False

if "session_results" not in st.session_state:
    st.session_state.session_results = None

if "session_feedback" not in st.session_state:
    st.session_state.session_feedback = []



# -------------------- CUSTOM CSS --------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #080d17;
    --panel: #111b2d;
    --panel-light: #162238;
    --border: #263650;
    --text: #f1f5f9;
    --muted: #94a3b8;
    --blue: #60a5fa;
    --purple: #818cf8;
}

.stApp {
    background: var(--bg);
    color: var(--text);
    font-family: 'Inter', sans-serif;
}

/* Hide Streamlit's top toolbar to prevent overlap */
[data-testid="stHeader"],
[data-testid="stToolbar"] {
    display: none !important;
}

/* Main page spacing */
.block-container {
    max-width: 1450px;
    padding: 2rem 2.2rem 3rem;
}

#MainMenu,
footer {
    visibility: hidden;
}

h1, h2, h3, p, label {
    font-family: 'Inter', sans-serif !important;
}

/* Brand header */
.brand {
    font-size: 24px;
    font-weight: 800;
    line-height: 1.5;
    padding-top: 8px;
    letter-spacing: -0.7px;
    color: #f8fafc;
    overflow: visible;
}

.brand span {
    color: #818cf8;
}

.brand-sub {
    color: #94a3b8;
    font-size: 11px;
    letter-spacing: 2px;
    margin-top: 5px;
}

.top-status {
    display: inline-block;
    padding: 8px 12px;
    border: 1px solid #34435f;
    background: #111b2d;
    color: #bfdbfe;
    border-radius: 30px;
    font-size: 12px;
    white-space: nowrap;
    margin-top: 8px;
}

/* Page heading */
.page-heading {
    font-size: 27px;
    font-weight: 750;
    letter-spacing: -0.7px;
    color: #f8fafc;
    margin: 24px 0 5px;
}

.page-description {
    font-size: 14px;
    line-height: 1.7;
    color: #94a3b8;
    margin-bottom: 22px;
}

/* Dashboard panels */
.panel {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 20px;
    margin-bottom: 16px;
}

.panel-heading {
    color: #f1f5f9;
    font-size: 16px;
    font-weight: 700;
    margin-bottom: 5px;
}

.panel-subtitle {
    color: var(--muted);
    font-size: 12px;
    line-height: 1.6;
    margin-bottom: 15px;
}

.eyebrow {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.8px;
    color: #a5b4fc;
    margin-bottom: 10px;
}

/* Reduce the live webcam preview size */
[data-testid="stWebRtcStreamer"] video {
    width: 100% !important;
    max-height: 320px !important;
    object-fit: contain !important;
}

/* Webcam placeholder */
.webcam-placeholder {
    background:
        radial-gradient(
            ellipse at center,
            #172b49 0%,
            #0a1220 72%
        );
    border: 1px solid #2a3b57;
    border-radius: 12px;
    min-height: 180px;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-direction: column;
    text-align: center;
    padding: 20px;
}

.camera-icon {
    font-size: 46px;
    margin-bottom: 10px;
}

.camera-title {
    color: #e2e8f0;
    font-size: 16px;
    font-weight: 700;
}

.camera-description {
    color: #94a3b8;
    font-size: 12px;
    margin-top: 7px;
    line-height: 1.6;
}

/* Overall score */
.score-wrap {
    text-align: center;
    padding: 15px 0 20px;
}

.score-ring {
    width: 150px;
    height: 150px;
    margin: 8px auto 12px;
    border-radius: 50%;
    border: 7px solid #34416b;
    box-shadow: 0 0 28px rgba(99, 102, 241, 0.12);
    display: flex;
    align-items: center;
    justify-content: center;
    flex-direction: column;
    background: #0c1424;
}

.score-number {
    font-size: 40px;
    line-height: 1.1;
    font-weight: 800;
    color: #f8fafc;
}

.score-denominator {
    color: #94a3b8;
    font-size: 12px;
    margin-top: 4px;
}

.score-caption {
    color: #94a3b8;
    font-size: 12px;
}

/* Component score cards */
.metric-card {
    background: var(--panel-light);
    border: 1px solid #2a3a55;
    border-radius: 11px;
    padding: 13px;
    min-height: 82px;
    margin-bottom: 10px;
}

.metric-label {
    font-size: 11px;
    color: #a9b8d0;
    margin-bottom: 8px;
}

.metric-value {
    font-size: 24px;
    font-weight: 750;
    color: #f8fafc;
}

/* Interview question */
.question-text {
    color: #f8fafc;
    font-size: 21px;
    font-weight: 700;
    line-height: 1.55;
    margin: 8px 0 12px;
    overflow-wrap: anywhere;
}

.question-hint {
    color: #94a3b8;
    font-size: 12px;
    line-height: 1.7;
}

.mini-tag {
    display: inline-block;
    color: #bfdbfe;
    background: #172b49;
    border: 1px solid #2a4568;
    border-radius: 6px;
    padding: 5px 8px;
    font-size: 10px;
    font-weight: 600;
}

/* Buttons */
.stButton > button {
    border-radius: 9px;
    min-height: 44px;
    font-weight: 700;
    border: 1px solid #6474e8;
    background: linear-gradient(110deg, #4f60d8, #7753df);
    color: white;
    transition: 0.2s ease;
}

.stButton > button:hover {
    color: white;
    border-color: #a5b4fc;
    filter: brightness(1.12);
}

.stButton > button:focus {
    color: white;
}

/* Dropdowns */
div[data-baseweb="select"] > div {
    background: #0c1424;
    border-color: #34435f;
    border-radius: 9px;
}

div[data-testid="stSelectbox"] label {
    color: #cbd5e1;
    font-weight: 600;
    font-size: 12px;
}

div[data-testid="stAlert"] {
    border-radius: 10px;
}

hr {
    border-color: #263650;
}

/* Footer */
.footer {
    text-align: center;
    color: #64748b;
    font-size: 11px;
    line-height: 1.8;
    padding-top: 20px;
}

/* Responsive adjustments */
@media (max-width: 700px) {
    .block-container {
        padding: 1.5rem 1rem 2rem;
    }

    .brand {
        font-size: 20px;
        line-height: 1.5;
        padding-top: 8px;
    }

    .brand-sub {
        font-size: 8px;
        letter-spacing: 1px;
    }

    .top-status {
        font-size: 9px;
        padding: 5px 7px;
        white-space: nowrap;
    }

    .page-heading {
        font-size: 23px;
        line-height: 1.3;
        margin-top: 20px;
    }

    .page-description {
        font-size: 13px;
    }

    .panel {
        padding: 15px;
        border-radius: 13px;
    }

    .webcam-placeholder {
        min-height: 230px;
        padding: 12px;
    }

    .question-text {
        font-size: 18px;
    }

    .score-ring {
        width: 125px;
        height: 125px;
    }

    .score-number {
        font-size: 34px;
    }

    .metric-card {
        padding: 11px;
    }

    .metric-value {
        font-size: 21px;
    }
}
</style>
""", unsafe_allow_html=True)


# -------------------- HEADER --------------------

header_left, header_right = st.columns([3, 1])

with header_left:
    st.markdown("""
    <div class="brand">Confident<span>AI</span></div>
    <div class="brand-sub">INTERVIEW PRACTICE STUDIO</div>
    """, unsafe_allow_html=True)

with header_right:
    st.markdown(
        '<div class="top-status">● &nbsp; Practice mode</div>',
        unsafe_allow_html=True
    )

st.markdown(
    '<div class="page-heading">Interview workspace</div>',
    unsafe_allow_html=True
)

st.markdown("""
<div class="page-description">
    Practice technical questions and review your behavioral indicators
    in one focused workspace.
</div>
""", unsafe_allow_html=True)


# -------------------- QUICK START --------------------

_, start_col, _ = st.columns([1, 2, 3])

with start_col:
    start_top = st.button(
        "▶  Start interview",
        key="start_interview_top",
        type="primary",
        use_container_width=True
    )


if start_top:
    # Reset measurements from the previous interview.
    session_metrics.reset()
    pose_history.clear()

    # Reset timer state for the new interview.
    st.session_state.interview_start_time = None
    st.session_state.time_expired = False


    # Clear previous report and feedback.
    st.session_state.session_results = None
    st.session_state.session_feedback = []

    # Start a fresh interview.
    st.session_state.interview_started = True
    st.session_state.interview_finished = False

    st.session_state.selected_question = st.session_state.get(
        "question_selector",
        "Explain the four pillars of OOP."
    )

    st.session_state.selected_duration = st.session_state.get(
        "duration_selector",
        60
    )

    st.rerun()



# -------------------- INTERVIEW TIMER --------------------



# -------------------- INTERVIEW TIMER --------------------



# -------------------- INTERVIEW TIMER --------------------
if st.session_state.interview_started:
    start_time = st.session_state.interview_start_time

    if start_time is None:
        st.info("📷 Start your webcam to begin the countdown.")
    else:
        duration = int(st.session_state.selected_duration)

        timer_html = f"""
        <div style="
            background:#111b2d;
            border:1px solid #34435f;
            border-radius:10px;
            padding:4px 12px;
            font-family:Arial,sans-serif;
            color:#f1f5f9;
            text-align:center;
            width:320px;
            max-width:90%;
            margin:30px 0 0 0;
            box-sizing:border-box;
        ">
            <div id="timer-title" style="font-size:12px;margin-bottom:2px;">
                INTERVIEW TIME REMAINING
            </div>
            <div id="timer-value" style="font-size:24px;font-weight:bold;">
                --:--
            </div>
            <div id="timer-message" style="font-size:10px;margin-top:2px;line-height:1.2; overflow-wrap:anywhere;">
            </div>
        </div>

        <script>
        const endTime = {start_time} + {duration};
        const timerValue = document.getElementById("timer-value");
        const timerTitle = document.getElementById("timer-title");
        const timerMessage = document.getElementById("timer-message");

        function updateTimer() {{
            const remaining = Math.max(
                0, Math.ceil(endTime - Date.now() / 1000)
            );

            const minutes = Math.floor(remaining / 60);
            const seconds = remaining % 60;

            timerValue.textContent =
                String(minutes).padStart(2, "0") + ":" +
                String(seconds).padStart(2, "0");

            if (remaining === 0) {{
                timerTitle.textContent = "INTERVIEW TIME OVER";
                timerMessage.textContent =
                    "Time Over!. Stop the webcam ";
                timerMessage.style.color = "#fca5a5";
                clearInterval(timerInterval);
            }}
        }}

        const timerInterval = setInterval(updateTimer, 250);
        updateTimer();
        </script>
        """

        components.html(timer_html, height=95)






# -------------------- MAIN DASHBOARD --------------------

left_column, right_column = st.columns(
    [1.55, 1],
    gap="medium"
)


# -------------------- LEFT COLUMN --------------------
with left_column:

    # Interview room heading
    st.markdown("""
    <div class="panel">
        <div class="panel-heading">▣ &nbsp; Interview room</div>
        <div class="panel-subtitle">
            Your live camera and facial-expression analysis
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Webcam must remain inside left_column
    if st.session_state.interview_started:
        ctx = webrtc_streamer(
            key="confidentai-interview",
            mode=WebRtcMode.SENDRECV,
            video_frame_callback=process_frame,
            media_stream_constraints={
                "video": {
                    "width": {"ideal": 480},
                    "height": {"ideal": 270},
                    "frameRate": {"ideal": 20, "max": 24}
                },
                "audio": False
            },
            rtc_configuration={
                "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]}
            ]
        },
            async_processing=True
        )

        if ctx.state.playing:
            if st.session_state.interview_start_time is None:
                st.session_state.interview_start_time = time.time()
                st.session_state.time_expired = False
                st.rerun()

        st.caption("Webcam active — your interview is in progress.")
            

    else:
        st.markdown("""
        <div class="webcam-placeholder">
            <div class="camera-icon">◉</div>
            <div class="camera-title">Your interview room</div>
            <div class="camera-description">
                Start your interview to activate the camera.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.info(
            "Your browser will request camera access when "
            "you start the interview."
        )

    # Question area comes after the webcam





    # Question area
    st.markdown("""
    <div class="panel">
        <div class="eyebrow">YOUR INTERVIEW QUESTION</div>
        <div class="panel-heading">Today's challenge</div>
    """, unsafe_allow_html=True)

    if st.session_state.get("reset_question", False):
        st.session_state.question_selector = (
        "Explain the four pillars of OOP."
        )
        st.session_state.selected_question = (
        "Explain the four pillars of OOP."
        )
        st.session_state.reset_question = False



    question = st.selectbox(
        "Choose a technical question",
        [
            "Explain the four pillars of OOP.",
            "What is the difference between SQL and NoSQL?",
            "Explain the difference between a process and a thread.",
            "What is a REST API?",
            "Explain the time complexity of binary search.",
            "What is the difference between an array and a linked list?",
            "Explain stack and queue with real-world examples.",
            "What is the difference between a primary key and a foreign key?",
            "Explain the difference between DELETE, DROP, and TRUNCATE in SQL.",
            "What is the difference between an interface and an abstract class in Java?",
            "Explain closures and hoisting in JavaScript.",
            "What is the difference between authentication and authorization?"
        ],
        key="question_selector",
        label_visibility="collapsed"
    )

    st.markdown(
        f'<div class="question-text">{question}</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
        <div class="question-hint">
            Take a moment to organize your thoughts. Explain the concept
            clearly and use an example where possible.
        </div>
    </div>
    """, unsafe_allow_html=True)


# -------------------- RIGHT COLUMN --------------------

with right_column:

    # Overall score
    
    results = st.session_state.session_results
    final_score = (
        results.get("final_score")
        if results else None
    )

    score_display = (
        f"{final_score:.1f}"
        if final_score is not None else "--"
    )

    score_caption = (
        "Analysis complete"
        if results and final_score is not None
        else "Awaiting interview analysis"
    )

    st.markdown(f"""
    <div class="panel">
        <div class="panel-heading">◎ &nbsp; Session score</div>
        <div class="panel-subtitle">
            Your overall result from this practice session
        </div>
        <div class="score-wrap">
            <div class="score-ring">
                <div class="score-number">{score_display}</div>
                <div class="score-denominator">out of 100</div>
            </div>
            <div class="score-caption">{score_caption}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


    # Behavioral indicators
    
    # Behavioral indicators
    results = st.session_state.session_results
    components = results.get("components", {}) if results else {}

    gaze = components.get("Estimated gaze")
    stability = components.get("Head stability")
    expression = components.get("Expression indicator")

    gaze_display = f"{gaze:.1f}" if gaze is not None else "--"
    stability_display = (
        f"{stability:.1f}" if stability is not None else "--"
    )
    expression_display = (
        f"{expression:.1f}" if expression is not None else "--"
    )

    st.markdown("""
    <div class="panel">
        <div class="panel-heading">Behavioral indicators</div>
        <div class="panel-subtitle">
            Component scores calculated from session measurements.
        </div>
    """, unsafe_allow_html=True)

    metric1, metric2 = st.columns(2)

    with metric1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">◉ &nbsp; Estimated gaze</div>
            <div class="metric-value">{gaze_display}</div>
        </div>
        """, unsafe_allow_html=True)

    with metric2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">↕ &nbsp; Head stability</div>
            <div class="metric-value">{stability_display}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">☺ &nbsp; Expression indicator</div>
        <div class="metric-value">{expression_display}</div>
    </div>
    </div>
    """, unsafe_allow_html=True)


    # Session setup
    st.markdown("""
    <div class="panel">
        <div class="panel-heading">⚙ &nbsp; Session setup</div>
        <div class="panel-subtitle">
            Choose the duration for this practice session.
        </div>
    """, unsafe_allow_html=True)

    def reset_duration():
        st.session_state.duration_selector = 60
        st.session_state.reset_question = True

    duration = st.selectbox(
        "Practice duration",
        [30, 60, 90, 120],
        index=1,
        format_func=lambda seconds: f"{seconds} seconds",
        key="duration_selector"
    )

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">PLANNED DURATION</div>
            <div class="metric-value">{duration}s</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    finish_clicked = st.button(
    "Finish interview & calculate score",
    key="finish_interview",
    use_container_width=True,
    disabled=not st.session_state.interview_started
    )   

    stop_clicked = st.button(
        "Reset session",
         key="reset_session",
        use_container_width=True,
        on_click=reset_duration
    )

    st.markdown("</div>", unsafe_allow_html=True)


# -------------------- BUTTON BEHAVIOR --------------------


# -------------------- BUTTON BEHAVIOR --------------------



if finish_clicked:
    summary = session_metrics.get_summary()

    score_result = calculate_confidence_score(
        [summary["gaze_score"]]
        if summary["gaze_score"] is not None else [],

        [summary["head_stability_score"]]
        if summary["head_stability_score"] is not None else [],

        [summary["expression_score"]]
        if summary["expression_score"] is not None else []
    )

    st.session_state.session_results = {
        **summary,
        **score_result
    }

    st.session_state.session_feedback = generate_feedback(
        score_result
    )

    st.session_state.interview_started = False
    st.session_state.interview_finished = True

    st.rerun()



if stop_clicked:
    st.session_state.reset_question = True
    # Keep the rest of your existing reset code below.
    # Stop the interview and clear its results.
    st.session_state.interview_started = False
    st.session_state.interview_finished = False

    # Reset timer state.
    st.session_state.interview_start_time = None
    st.session_state.time_expired = False

    # Clear the displayed report.
    st.session_state.session_results = None
    st.session_state.session_feedback = []

    # Clear collected measurements and head-pose history.
    session_metrics.reset()
    pose_history.clear()

    # Restore default selections.
    st.session_state.selected_question = (
        "Explain the four pillars of OOP."
    )
    st.session_state.selected_duration = 60

    st.rerun()


if st.session_state.interview_started:
    st.success("Interview session started.")

    st.write(
        f"**Question:** {st.session_state.selected_question}"
    )

    st.write(
        f"**Duration:** {st.session_state.selected_duration} seconds"
    )



# -------------------- FEEDBACK PANEL --------------------


# -------------------- FEEDBACK PANEL --------------------

st.markdown("---")
st.markdown("""
<div class="panel">
    <div class="panel-heading">✦ &nbsp; Feedback and insights</div>
    <div class="panel-subtitle">
        Review your detected emotions and behavioral indicators.
    </div>
</div>
""", unsafe_allow_html=True)

results = st.session_state.session_results

if results:
    dominant_emotion = results.get("dominant_emotion")
    emotion_distribution = results.get("emotion_distribution", {})

    if dominant_emotion:
        st.markdown(f"**Most detected emotion:** {dominant_emotion}")
    else:
        st.info("No emotion was detected during this session.")

    st.markdown("**Emotion distribution**")

    if emotion_distribution:
        st.bar_chart(emotion_distribution)
    else:
        st.info("No emotion data is available yet.")

    st.markdown("**Personalized feedback**")

    feedback = st.session_state.session_feedback

    if feedback:
        for item in feedback:
            st.write(f"• {item}")
    else:
        st.info("There is not enough data to generate feedback.")

elif st.session_state.interview_finished:
    st.info("The session finished, but no results are available.")

else:
    st.info(
        "Finish an interview to see your emotion distribution "
        "and personalized feedback here."
    )



# -------------------- FOOTER --------------------

st.markdown("""
<div class="footer">
    © 2026 ConfidentAI · Interview Practice Studio<br>
    Practice smarter. Prepare with confidence.
</div>
""", unsafe_allow_html=True)
