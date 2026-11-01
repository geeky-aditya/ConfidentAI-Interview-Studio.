
import cv2
import numpy as np


EMOTIONS = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]


def predict_emotion(face, model):
    """
    Predict facial expression using the trained CNN.

    face: BGR face crop from OpenCV.
    model: loaded emotion_cnn.keras model.
    """

    if face is None or face.size == 0:
        return None

    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (48, 48))
    gray = gray.astype(np.float32) / 255.0

    input_face = np.expand_dims(gray, axis=(0, -1))

    probabilities = model.predict(
        input_face,
        verbose=0
    )[0]

    index = int(np.argmax(probabilities))

    return {
        "emotion": EMOTIONS[index],
        "emotion_probability": float(probabilities[index]),
        "probabilities": probabilities
    }


def calculate_expression_indicator(probabilities):
    """
    Experimental expression indicator based on the
    CNN probabilities for Happy and Neutral.

    This is NOT a measure of actual interview confidence.
    """

    probabilities = np.asarray(probabilities)

    return float(
        (probabilities[3] + probabilities[4]) * 50.0
    )
