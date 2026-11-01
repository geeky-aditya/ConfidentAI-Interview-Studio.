
import cv2


def open_camera(camera_index=0):
    """
    Open the webcam and return the camera object.
    """
    camera = cv2.VideoCapture(camera_index)

    if not camera.isOpened():
        camera.release()
        raise RuntimeError(
            "Could not open webcam. Check camera permissions."
        )

    return camera


def read_frame(camera):
    """
    Read one frame from the webcam.
    Returns None if reading fails.
    """
    success, frame = camera.read()

    if not success:
        return None

    # Mirror the webcam image
    frame = cv2.flip(frame, 1)

    return frame


def release_camera(camera):
    """
    Release the webcam and close OpenCV windows.
    """
    if camera is not None:
        camera.release()

    cv2.destroyAllWindows()
