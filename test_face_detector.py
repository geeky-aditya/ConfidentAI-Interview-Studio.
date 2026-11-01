
import cv2

from webcam.camera import (
    open_camera,
    read_frame,
    release_camera
)

from webcam.face_detector import FaceLandmarkDetector


camera = None
detector = None

try:
    camera = open_camera()
    detector = FaceLandmarkDetector()

    print("Face landmark detector loaded!")
    print("Press Q to exit.")

    while True:
        frame = read_frame(camera)

        if frame is None:
            print("Could not read frame.")
            break

        result = detector.detect(frame)

        if result is not None:
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
                f"Landmarks: {len(result['points'])}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        else:
            cv2.putText(
                frame,
                "No face detected",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        cv2.imshow("Face Detector Test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    release_camera(camera)

    if detector is not None:
        detector.close()

print("Face detection test completed.")
