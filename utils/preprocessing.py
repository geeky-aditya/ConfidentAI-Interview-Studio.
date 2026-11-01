
import cv2


def prepare_rgb_frame(frame):
    """
    Convert an OpenCV BGR frame to RGB.
    """
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def crop_face(frame, bbox):
    """
    Crop a face from the frame using:
    bbox = (x1, y1, x2, y2)
    """

    if frame is None or bbox is None:
        return None

    x1, y1, x2, y2 = bbox

    height, width = frame.shape[:2]

    # Keep coordinates inside the frame
    x1 = max(0, min(int(x1), width))
    x2 = max(0, min(int(x2), width))
    y1 = max(0, min(int(y1), height))
    y2 = max(0, min(int(y2), height))

    if x2 <= x1 or y2 <= y1:
        return None

    return frame[y1:y2, x1:x2].copy()
