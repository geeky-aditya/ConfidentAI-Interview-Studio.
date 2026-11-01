
import cv2

from webcam.camera import (
    open_camera,
    read_frame,
    release_camera
)

from webcam.face_detector import FaceLandmarkDetector
from webcam.head_pose import estimate_head_pose


camera = None
detector = None

try:
    camera = open_camera()
    detector = FaceLandmarkDetector()

    print("Head pose test started.")
    print("Move your head slowly.")
    print("Press Q to exit.")

    while True:
        frame = read_frame(camera)

        if frame is None:
            break

        result = detector.detect(frame)

        if result is not None:
            pose = estimate_head_pose(
                result["points"],
                frame.shape,
                result["transform"]
            )

            x1, y1, x2, y2 = result["bbox"]

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            if pose is not None:
                label = (
                    f"Pitch: {pose['pitch']:.1f}  "
                    f"Yaw: {pose['yaw']:.1f}  "
                    f"Roll: {pose['roll']:.1f}"
                )

                cv2.putText(
                    frame,
                    label,
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2
                )

        cv2.imshow("Head Pose Test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    release_camera(camera)

    if detector is not None:
        detector.close()

print("Head pose test completed.")
