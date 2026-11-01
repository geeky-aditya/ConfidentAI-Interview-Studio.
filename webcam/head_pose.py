
import numpy as np


def estimate_head_pose(points, frame_shape, transform=None):
    """
    Estimate approximate head orientation in degrees.

    Uses MediaPipe's facial transformation matrix when available.
    Returns pitch, yaw, and roll, or None if unavailable.
    """

    if transform is None:
        return None

    matrix = np.asarray(transform, dtype=np.float64)

    # MediaPipe generally returns a 4x4 transformation matrix.
    if matrix.size != 16:
        return None

    matrix = matrix.reshape(4, 4)
    rotation = matrix[:3, :3]

    # Remove scale effects with SVD to obtain a rotation matrix.
    u, _, vt = np.linalg.svd(rotation)
    rotation = u @ vt

    # Ensure a proper rotation matrix.
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt

    sy = np.sqrt(
        rotation[0, 0] ** 2 + rotation[1, 0] ** 2
    )

    if sy > 1e-6:
        x_angle = np.arctan2(
            rotation[2, 1], rotation[2, 2]
        )
        y_angle = np.arctan2(
            -rotation[2, 0], sy
        )
        z_angle = np.arctan2(
            rotation[1, 0], rotation[0, 0]
        )
    else:
        x_angle = np.arctan2(
            -rotation[1, 2], rotation[1, 1]
        )
        y_angle = np.arctan2(
            -rotation[2, 0], sy
        )
        z_angle = 0.0

    return {
        "pitch": float(np.degrees(x_angle)),
        "yaw": float(np.degrees(y_angle)),
        "roll": float(np.degrees(z_angle))
    }


def calculate_head_stability(pose_history):
    """
    Calculate an experimental head-orientation consistency score.
    """

    if len(pose_history) < 2:
        return None

    angles = np.array([
        [p["pitch"], p["yaw"], p["roll"]]
        for p in pose_history
    ])

    # Unwrap angles so crossing +/-180 degrees doesn't
    # create a falsely huge movement.
    angles = np.degrees(np.unwrap(np.radians(angles), axis=0))

    variability = float(np.mean(np.std(angles, axis=0)))

    score = 100.0 * np.exp(-variability / 12.0)

    return round(float(np.clip(score, 0, 100)), 2)
