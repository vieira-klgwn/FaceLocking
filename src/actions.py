# src/actions.py
"""
Action detection: blink, smile, face moved left/right.
Supports: (1) MediaPipe 468-point landmarks, (2) 5-point fallback.
"""

from typing import List, Tuple, Union
import numpy as np


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


class ActionDetector5pt:
    """
    Uses 5-point landmarks [Leye, Reye, Nose, Lmouth, Rmouth].
    Detects: face moved left/right. (Smile and blink need MediaPipe Face Mesh.)
    """

    def __init__(self):
        self.MOVE_THRESHOLD = 6.0  # pixels
        self._last_centroid: np.ndarray | None = None

    def process_5pt(self, kps: np.ndarray) -> List[Tuple[str, str]]:
        """kps: (5,2) [Leye, Reye, Nose, Lmouth, Rmouth]"""
        out: List[Tuple[str, str]] = []
        centroid = np.mean(kps, axis=0)
        if self._last_centroid is not None:
            dx = centroid[0] - self._last_centroid[0]
            if abs(dx) > self.MOVE_THRESHOLD:
                direction = "right" if dx > 0 else "left"
                out.append(("move", f"face moved {direction}"))
        self._last_centroid = centroid.copy()
        return out


class ActionDetector:
    """
    Detects facial actions from 468-point MediaPipe landmarks.
    """

    def __init__(self):
        self.EAR_THRESHOLD = 0.22  # Eye Aspect Ratio for blink
        self.SMILE_RATIO_THRESHOLD = 0.45  # Mouth width/eye_dist for smile
        self.MOVE_THRESHOLD = 0.015  # Normalized movement for left/right

        self._blinking = False
        self._last_centroid: np.ndarray | None = None

    def _dist(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a - b))

    def _ear(self, landmarks: np.ndarray) -> float:
        """Eye Aspect Ratio (lower = more closed)."""
        # Left eye: 159-145 (vert), 33-133 (horiz)
        # Right eye: 386-374 (vert), 362-263 (horiz)
        v1 = self._dist(landmarks[159], landmarks[145])
        h1 = self._dist(landmarks[33], landmarks[133])
        ear_l = v1 / (h1 + 1e-6)

        v2 = self._dist(landmarks[386], landmarks[374])
        h2 = self._dist(landmarks[362], landmarks[263])
        ear_r = v2 / (h2 + 1e-6)
        return (ear_l + ear_r) / 2.0

    def _mouth_ratio(self, landmarks: np.ndarray) -> float:
        """Mouth width / eye distance as smile indicator."""
        w = self._dist(landmarks[61], landmarks[291])
        eye_dist = self._dist(landmarks[33], landmarks[263])
        return w / (eye_dist + 1e-6)

    def process(self, landmarks: np.ndarray) -> List[Tuple[str, str]]:
        """
        Detect actions from landmarks.
        Returns list of (action_type, description).
        """
        out: List[Tuple[str, str]] = []

        # Blink
        ear = self._ear(landmarks)
        if ear < self.EAR_THRESHOLD:
            if not self._blinking:
                self._blinking = True
                out.append(("blink", f"blink (EAR={ear:.2f})"))
        else:
            self._blinking = False

        # Smile
        mr = self._mouth_ratio(landmarks)
        if mr > self.SMILE_RATIO_THRESHOLD:
            out.append(("smile", f"smile/laugh (ratio={mr:.2f})"))

        # Movement left/right (use nose or centroid)
        centroid = np.mean(landmarks[:, :2], axis=0)
        if self._last_centroid is not None:
            dx = centroid[0] - self._last_centroid[0]
            if abs(dx) > self.MOVE_THRESHOLD:
                direction = "right" if dx > 0 else "left"
                out.append(("move", f"face moved {direction}"))
        self._last_centroid = centroid

        return out
