
import cv2

from webcam.camera import (
    open_camera,
    read_frame,
    release_camera
)

camera = None

try:
    camera = open_camera()

    print("Camera opened successfully!")
    print("Press Q in the camera window to exit.")

    while True:
        frame = read_frame(camera)

        if frame is None:
            print("Could not read webcam frame.")
            break

        cv2.imshow("Camera Test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    release_camera(camera)

print("Camera test completed.")
