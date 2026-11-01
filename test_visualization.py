
import cv2

from utils.visualization import (
    draw_face_box,
    draw_text,
    draw_landmarks
)

frame = cv2.imread("accuracy_graph.png")

if frame is None:
    raise FileNotFoundError("Could not load accuracy_graph.png")

height, width = frame.shape[:2]

draw_face_box(
    frame,
    (20, 20, width - 20, height - 20),
    "Visualization Test"
)

draw_text(
    frame,
    "Drawing works!",
    (30, 60)
)

draw_landmarks(
    frame,
    [[100, 100], [150, 150], [200, 100]]
)

cv2.imshow("Visualization Test", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Visualization test passed!")
