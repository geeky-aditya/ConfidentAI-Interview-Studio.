
import cv2


def draw_face_box(frame, bbox, label=None):
    """
    Draw a rectangle around the detected face
    and optionally display a label.
    
    bbox format: (x1, y1, x2, y2)
    """

    if frame is None or bbox is None:
        return frame

    x1, y1, x2, y2 = bbox

    cv2.rectangle(
        frame,
        (int(x1), int(y1)),
        (int(x2), int(y2)),
        (0, 255, 0),
        2
    )

    if label:
        cv2.putText(
            frame,
            str(label),
            (int(x1), max(25, int(y1) - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

    return frame


def draw_text(
    frame,
    text,
    position=(20, 40),
    color=(0, 255, 0),
    font_scale=0.7
):
    """
    Display text on a webcam frame.
    """

    if frame is None:
        return frame

    cv2.putText(
        frame,
        str(text),
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        2,
        cv2.LINE_AA
    )

    return frame


def draw_landmarks(frame, points, color=(0, 255, 255)):
    """
    Draw detected facial landmark points.
    Useful for debugging eye and head-pose analysis.
    """

    if frame is None or points is None:
        return frame

    for point in points:
        x, y = int(point[0]), int(point[1])

        cv2.circle(
            frame,
            (x, y),
            1,
            color,
            -1
        )

    return frame
