
import os
import cv2
import mediapipe as mp
import numpy as np


# Path to MediaPipe Face Landmarker model
MODEL_PATH = os.path.join(
    "assets",
    "face_landmarker.task"
)


class FaceLandmarkDetector:

    def __init__(self):
        # Check whether model file exists
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Face landmark model not found: {MODEL_PATH}"
            )

        # Configure MediaPipe Tasks API
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=MODEL_PATH
            ),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True
        )

        # Create detector
        self.detector = (
            mp.tasks.vision.FaceLandmarker
            .create_from_options(options)
        )

        print("MediaPipe Face Landmarker loaded successfully!")

    def detect(self, frame):
        """
        Detect a face and return its landmarks,
        bounding box, blendshapes, and transformation matrix.
        """

        if frame is None:
            return None

        # OpenCV uses BGR; MediaPipe expects SRGB/RGB
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Run face landmark detection
        result = self.detector.detect(mp_image)

        # No face found
        if not result.face_landmarks:
            return None

        # First detected face
        landmarks = result.face_landmarks[0]

        height, width = frame.shape[:2]

        # Convert normalized landmarks into pixel coordinates
        points = np.array([
            [
                int(landmark.x * width),
                int(landmark.y * height)
            ]
            for landmark in landmarks
        ])

        # Calculate bounding box
        x_min = max(0, int(points[:, 0].min()))
        y_min = max(0, int(points[:, 1].min()))
        x_max = min(width - 1, int(points[:, 0].max()))
        y_max = min(height - 1, int(points[:, 1].max()))

        # Return all useful face information
        return {
            "landmarks": landmarks,
            "points": points,
            "bbox": (x_min, y_min, x_max, y_max),

            "blendshapes": (
                result.face_blendshapes[0]
                if result.face_blendshapes
                else None
            ),

            "transform": (
                result.facial_transformation_matrixes[0]
                if result.facial_transformation_matrixes
                else None
            )
        }

    def close(self):
        """Release MediaPipe resources."""
        if self.detector is not None:
            self.detector.close()
