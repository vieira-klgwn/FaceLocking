# src/landmarks_mp.py
"""
MediaPipe Face Mesh for action detection (blinks, smiles).
Returns 468 landmarks in normalized [0,1] coordinates.
"""

from typing import Optional, Tuple
import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError:
    mp = None


class MediaPipeLandmarker:
    """
    High-resolution facial landmarks from MediaPipe Face Mesh.
    Used for EAR (blink) and mouth-ratio (smile) detection.
    """

    def __init__(
        self,
        static_mode: bool = False,
        max_faces: int = 1,
        refine_landmarks: bool = True,
    ):
        if mp is None:
            raise RuntimeError("mediapipe not installed. Run: pip install mediapipe")
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=static_mode,
            max_num_faces=max_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def get_landmarks(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract landmarks from frame.
        Can use full frame or cropped face region.
        Returns (N, 3) array [x, y, z] normalized [0, 1].
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            return None
        lm = results.multi_face_landmarks[0]
        pts = [[p.x, p.y, p.z] for p in lm.landmark]
        return np.array(pts, dtype=np.float32)

    @staticmethod
    def to_pixel_coords(landmarks: np.ndarray, h: int, w: int) -> np.ndarray:
        """Convert normalized landmarks to pixel coordinates."""
        out = landmarks.copy()
        out[:, 0] *= w
        out[:, 1] *= h
        return out
