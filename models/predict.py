import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import os


# ==========================================
# 1. LOAD CNN MODEL
# ==========================================

MODEL_PATH = "models/emotion_cnn.keras"

model = tf.keras.models.load_model(MODEL_PATH)

print("Emotion CNN model loaded.")


# ==========================================
# 2. EMOTION LABELS
# ==========================================

EMOTIONS = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]


# ==========================================
# 3. LOAD MEDIAPIPE FACE DETECTOR
# ==========================================

FACE_MODEL_PATH = "assets/blaze_face_short_range.tflite"

if not os.path.exists(FACE_MODEL_PATH):
    print("ERROR: MediaPipe face model not found.")
    exit()

BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
RunningMode = mp.tasks.vision.RunningMode

options = FaceDetectorOptions(
    base_options=BaseOptions(
        model_asset_path=FACE_MODEL_PATH
    ),
    running_mode=RunningMode.IMAGE,
    min_detection_confidence=0.5
)

detector = FaceDetector.create_from_options(options)

print("MediaPipe face detector loaded.")


# ==========================================
# 4. START WEBCAM
# ==========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

print("Webcam started.")
print("Answer your interview question.")
print("Press Q when you want to finish.")
print("----------------------------------------")


# ==========================================
# 5. STORE PREDICTIONS
# ==========================================

all_predictions = []

frame_count = 0


# ==========================================
# 6. WEBCAM LOOP
# ==========================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read webcam frame.")
        break

    # Mirror webcam
    frame = cv2.flip(frame, 1)

    height, width, _ = frame.shape

    # --------------------------------------
    # Convert BGR → RGB
    # --------------------------------------

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # --------------------------------------
    # Detect face
    # --------------------------------------

    result = detector.detect(mp_image)

    if len(result.detections) > 0:

        detection = result.detections[0]

        bbox = detection.bounding_box

        x = max(0, bbox.origin_x)
        y = max(0, bbox.origin_y)

        x2 = min(width, x + bbox.width)
        y2 = min(height, y + bbox.height)

        # Draw face rectangle
        cv2.rectangle(
            frame,
            (x, y),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # ----------------------------------
        # Crop face
        # ----------------------------------

        face = frame[y:y2, x:x2]

        if face.size != 0:

            # Grayscale
            gray_face = cv2.cvtColor(
                face,
                cv2.COLOR_BGR2GRAY
            )

            # Resize to 48x48
            gray_face = cv2.resize(
                gray_face,
                (48, 48)
            )

            # Normalize
            gray_face = gray_face.astype(
                "float32"
            ) / 255.0

            # Add dimensions
            input_face = np.expand_dims(
                gray_face,
                axis=0
            )

            input_face = np.expand_dims(
                input_face,
                axis=-1
            )

            # ----------------------------------
            # CNN PREDICTION
            # ----------------------------------

            predictions = model.predict(
                input_face,
                verbose=0
            )[0]

            # Store prediction
            all_predictions.append(predictions)

            frame_count += 1

            # Current emotion for webcam display
            emotion_index = np.argmax(predictions)

            emotion = EMOTIONS[emotion_index]

            emotion_confidence = (
                predictions[emotion_index] * 100
            )

            # ----------------------------------
            # DISPLAY CURRENT EMOTION
            # ----------------------------------

            cv2.putText(
                frame,
                f"Emotion: {emotion}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Confidence: {emotion_confidence:.1f}%",
                (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                "Press Q to finish",
                (20, height - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

    else:

        cv2.putText(
            frame,
            "No face detected",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    # Show webcam
    cv2.imshow(
        "Emotion Recognition",
        frame
    )

    # --------------------------------------
    # PRESS Q TO FINISH
    # --------------------------------------

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ==========================================
# 7. CLEANUP
# ==========================================

cap.release()
detector.close()
cv2.destroyAllWindows()


# ==========================================
# 8. FINAL SCORE
# ==========================================

print("\n")
print("========================================")
print("       INTERVIEW ANALYSIS COMPLETE")
print("========================================")


if len(all_predictions) == 0:

    print("No face was detected.")
    print("Unable to calculate emotion score.")

else:

    # Convert list to numpy array
    all_predictions = np.array(all_predictions)

    # Average predictions from entire session
    average_prediction = np.mean(
        all_predictions,
        axis=0
    )

    # Final dominant emotion
    final_emotion_index = np.argmax(
        average_prediction
    )

    final_emotion = EMOTIONS[
        final_emotion_index
    ]

    # Final emotion confidence
    final_confidence = (
        average_prediction[final_emotion_index]
        * 100
    )

    print(f"Frames analyzed: {frame_count}")
    print(f"Dominant Emotion: {final_emotion}")
    print(
        f"Final Emotion Score: "
        f"{final_confidence:.2f}%"
    )

    print("----------------------------------------")

    print("Emotion Distribution:")

    for i in range(len(EMOTIONS)):

        percentage = (
            average_prediction[i] * 100
        )

        print(
            f"{EMOTIONS[i]:10s}: "
            f"{percentage:.2f}%"
        )

print("========================================")
print("Program ended.")