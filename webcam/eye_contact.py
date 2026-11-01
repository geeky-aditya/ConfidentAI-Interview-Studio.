
import numpy as np


def calculate_eye_contact(points):
    """
    Estimate how centrally the irises are positioned
    within the eyes.

    points: NumPy array of MediaPipe landmark pixel
            coordinates with shape (478, 2).

    Returns:
        Estimated gaze score from 0 to 100,
        or None if the landmarks are unavailable.

    Note:
        This is a rough gaze-direction heuristic,
        not a precise eye-tracking measurement.
    """

    points = np.asarray(points)

    if points.ndim != 2 or points.shape[0] < 478:
        return None

    # Eye corner A, eye corner B, iris center
    eye_landmarks = [
        (33, 133, 468),
        (362, 263, 473)
    ]

    scores = []

    for corner_a, corner_b, iris_center in eye_landmarks:

        left_x = float(points[corner_a][0])
        right_x = float(points[corner_b][0])
        iris_x = float(points[iris_center][0])

        eye_width = abs(right_x - left_x)

        if eye_width < 1:
            continue

        # Position of iris relative to the eye corners
        ratio = abs(iris_x - left_x) / eye_width

        # Approximately centered when ratio is 0.5
        deviation = abs(ratio - 0.5)

        # Convert centeredness to a score from 0 to 100
        score = max(
            0.0,
            1.0 - deviation / 0.5
        ) * 100

        scores.append(score)

    if not scores:
        return None

    return round(float(np.mean(scores)), 2)
