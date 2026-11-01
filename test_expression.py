
import cv2
import tensorflow as tf

from webcam.camera import (
    open_camera,
    read_frame,
    release_camera
)

from webcam.face_detector import FaceLandmarkDetector
from utils.preprocessing import crop_face
from webcam.expression import predict_emotion


MODEL_PATH = "models/emotion_cnn.keras"

camera = None
detector = None

try:
    model = tf.keras.models.load_model(MODEL_PATH)

    camera = open_camera()
    detector = FaceLandmarkDetector()

    print("CNN model loaded.")
    print("Expression test started. Press Q to exit.")

    while True:
        frame = read_frame(camera)

        if frame is None:
            break

        result = detector.detect(frame)

        if result is not None:
            face = crop_face(frame, result["bbox"])

            prediction = predict_emotion(face, model)

            if prediction is not None:
                emotion = prediction["emotion"]
                probability = (
                    prediction["emotion_probability"] * 100
                )

                x1, y1, x2, y2 = result["bbox"]

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"{emotion}: {probability:.1f}%",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

        cv2.imshow("Expression Test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    release_camera(camera)

    if detector is not None:
        detector.close()

print("Expression test completed.")
