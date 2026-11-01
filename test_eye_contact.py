
import cv2

from webcam.camera import (
    open_camera,
    read_frame,
    release_camera
)

from webcam.face_detector import FaceLandmarkDetector
from webcam.eye_contact import calculate_eye_contact


camera = None
detector = None

try:
    camera = open_camera()
    detector = FaceLandmarkDetector()

    print("Eye-contact test started.")
    print("Look toward the camera.")
    print("Press Q to finish.")

    while True:
        frame = read_frame(camera)

        if frame is None:
            break

        result = detector.detect(frame)

        if result is not None:
            points = result["points"]

            gaze_score = calculate_eye_contact(points)

            x1, y1, x2, y2 = result["bbox"]

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            if gaze_score is not None:
                label = f"Estimated Gaze Score: {gaze_score:.1f}"

                cv2.putText(
                    frame,
                    label,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

        else:
            cv2.putText(
                frame,
                "No face detected",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

        cv2.imshow("Eye Contact Test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    release_camera(camera)

    if detector is not None:
        detector.close()

print("Eye-contact test completed.")
