
import os
import threading
import av
import cv2
import mediapipe as mp
import tensorflow as tf
import numpy as np

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from webcam.expression import (
    predict_emotion,
    calculate_expression_indicator
)
from webcam.eye_contact import calculate_eye_contact
from webcam.head_pose import (
    estimate_head_pose,
    calculate_head_stability
)
from webcam.session_metrics import SessionMetrics


# --------------------------------------------------
# PATHS
# --------------------------------------------------

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CNN_PATH = os.path.join(ROOT, "models", "emotion_cnn.keras")
FACE_PATH = os.path.join(
    ROOT, "assets", "blaze_face_short_range.tflite"
)
LANDMARKER_PATH = os.path.join(
    ROOT, "assets", "face_landmarker.task"
)


# --------------------------------------------------
# CHECK MODEL FILES
# --------------------------------------------------

for path in [CNN_PATH, FACE_PATH, LANDMARKER_PATH]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")


# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------

emotion_model = tf.keras.models.load_model(
    CNN_PATH,
    compile=False
)

detector_options = vision.FaceDetectorOptions(
    base_options=python.BaseOptions(
        model_asset_path=FACE_PATH
    ),
    running_mode=vision.RunningMode.IMAGE,
    min_detection_confidence=0.5
)

face_detector = vision.FaceDetector.create_from_options(
    detector_options
)

landmarker_options = vision.FaceLandmarkerOptions(
    base_options=python.BaseOptions(
        model_asset_path=LANDMARKER_PATH
    ),
    running_mode=vision.RunningMode.IMAGE,
    num_faces=1,
    output_facial_transformation_matrixes=True
)

face_landmarker = vision.FaceLandmarker.create_from_options(
    landmarker_options
)


# --------------------------------------------------
# SESSION DATA
# --------------------------------------------------

session_metrics = SessionMetrics()
pose_history = []
processing_lock = threading.Lock()

print("ConfidentAI processor loaded successfully.")


# --------------------------------------------------
# PROCESS A WEBCAM FRAME
# --------------------------------------------------

def process_frame(video_frame):
    frame = video_frame.to_ndarray(format="bgr24")

    with processing_lock:
        try:
            height, width = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            # Detect face bounding box.
            detection_result = face_detector.detect(mp_image)

            if not detection_result.detections:
                cv2.putText(
                    frame,
                    "No face detected",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

                return av.VideoFrame.from_ndarray(
                    frame, format="bgr24"
                )

            detection = max(
                detection_result.detections,
                key=lambda item: (
                    item.bounding_box.width
                    * item.bounding_box.height
                )
            )

            bbox = detection.bounding_box

            x1 = max(0, bbox.origin_x)
            y1 = max(0, bbox.origin_y)
            x2 = min(width, x1 + bbox.width)
            y2 = min(height, y1 + bbox.height)

            face = frame[y1:y2, x1:x2]

            if face.size == 0:
                return av.VideoFrame.from_ndarray(
                    frame, format="bgr24"
                )

            # --------------------------------------------------
            # 1. EMOTION DETECTION
            # --------------------------------------------------

            prediction = predict_emotion(face, emotion_model)

            if prediction is not None:
                emotion = prediction["emotion"]
                probabilities = prediction["probabilities"]

                expression_score = calculate_expression_indicator(
                    probabilities
                )

                session_metrics.record_sample(
                    expression_score=expression_score,
                    emotion=emotion
                )

                probability = (
                    prediction["emotion_probability"] * 100
                )

                

            # --------------------------------------------------
            # 2. FACIAL LANDMARKS
            # --------------------------------------------------

            landmark_result = face_landmarker.detect(mp_image)

            if landmark_result.face_landmarks:
                landmarks = landmark_result.face_landmarks[0]

                # Convert normalized landmarks into pixel coordinates.
                points = np.array([
                    [point.x * width, point.y * height, point.z * width]
                    for point in landmarks
                ], dtype=np.float64)

                # --------------------------------------------------
                # 3. ESTIMATED EYE CONTACT
                # --------------------------------------------------

                gaze_score = calculate_eye_contact(points)

                # --------------------------------------------------
                # 4. HEAD POSE AND STABILITY
                # --------------------------------------------------

                head_pose = None

                matrices = (
                    landmark_result.facial_transformation_matrixes
                )

                if matrices:
                    head_pose = estimate_head_pose(
                        points,
                        frame.shape,
                        matrices[0]
                    )

                head_stability_score = None

                if head_pose is not None:
                    pose_history.append(head_pose)

                    # Keep recent history bounded.
                    if len(pose_history) > 3000:
                        del pose_history[:1000]

                    head_stability_score = calculate_head_stability(
                        pose_history
                    )

                # Record measurements that were available this frame.
                session_metrics.record_sample(
                    gaze_score=gaze_score,
                    head_stability_score=head_stability_score
                )

                # Show landmark dots for visual verification.
                for point in points[::10]:
                    px = int(point[0])
                    py = int(point[1])

                    if 0 <= px < width and 0 <= py < height:
                        cv2.circle(
                            frame,
                            (px, py),
                            1,
                            (255, 180, 0),
                            -1
                        )

                

               

            # Draw face rectangle.
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 100),
                2
            )

            return av.VideoFrame.from_ndarray(
                frame, format="bgr24"
            )

        except Exception as error:
            print(f"Frame processing error: {error}")

            return av.VideoFrame.from_ndarray(
                frame, format="bgr24"
            )
