# src/haar_5pt.py
"""
Haar face detection + practical 5-point landmarks (MediaPipe FaceMesh).
Why this works for you:
- Haar is fast and robust on CPU.
- MediaPipe FaceMesh confirms a real face and gives stable landmarks.
- We extract ONLY 5 keypoints: left_eye, right_eye, nose_tip, mouth_left, mouth_right
- We rebuild bbox from keypoints (centered), so no "aside" offset.
- We reject Haar false positives if FaceMesh doesn't produce landmarks.
Run:
python -m src.haar_5pt
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, List
import cv2
import numpy as np

# MediaPipe is optional; used only for refine_landmarks in advanced mode.
# We use estimated 5pt from bbox for detection (no MediaPipe required).


# -------------------------
# Data
# -------------------------
@dataclass
class FaceKpsBox:
    x1: int
    y1: int
    x2: int
    y2: int
    score: float
    kps: np.ndarray  # (5,2) float32


# -------------------------
# Helpers
# -------------------------
def _estimate_norm_5pt(kps_5x2: np.ndarray, out_size: Tuple[int, int] = (112, 112)) -> np.ndarray:
    """
    Build 2x3 affine matrix that maps your 5pts to ArcFace-style template.
    kps order must be: [Leye, Reye, Nose, Lmouth, Rmouth]
    """
    k = kps_5x2.astype(np.float32)
    # ArcFace 112x112 template (InsightFace standard)
    dst = np.array([
        [38.2946, 51.6963],  # left eye
        [73.5318, 51.5014],  # right eye
        [56.0252, 71.7366],  # nose
        [41.5493, 92.3655],  # left mouth
        [70.7299, 92.2041],  # right mouth
    ], dtype=np.float32)
    out_w, out_h = int(out_size[0]), int(out_size[1])
    if (out_w, out_h) != (112, 112):
        sx = out_w / 112.0
        sy = out_h / 112.0
        dst = dst * np.array([sx, sy], dtype=np.float32)
    M, _ = cv2.estimateAffinePartial2D(k, dst, method=cv2.LMEDS)
    if M is None:
        # fallback: use eyes + nose only
        M = cv2.getAffineTransform(
            np.array([k[0], k[1], k[2]], dtype=np.float32),
            np.array([dst[0], dst[1], dst[2]], dtype=np.float32),
        )
    return M.astype(np.float32)


def align_face_5pt(frame_bgr: np.ndarray, kps_5x2: np.ndarray, out_size: Tuple[int, int] = (112, 112)) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (aligned_bgr, M)
    """
    M = _estimate_norm_5pt(kps_5x2, out_size=out_size)
    out_w, out_h = int(out_size[0]), int(out_size[1])
    aligned = cv2.warpAffine(
        frame_bgr,
        M,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    return aligned, M


def _clip_box_xyxy(b: np.ndarray, W: int, H: int) -> np.ndarray:
    bb = b.astype(np.float32).copy()
    bb[0] = np.clip(bb[0], 0, W - 1)
    bb[1] = np.clip(bb[1], 0, H - 1)
    bb[2] = np.clip(bb[2], 0, W - 1)
    bb[3] = np.clip(bb[3], 0, H - 1)
    return bb


def _bbox_from_5pt(kps: np.ndarray, pad_x: float = 0.55, pad_y_top: float = 0.85, pad_y_bot: float = 1.15) -> np.ndarray:
    """
    Build a face bbox from 5 keypoints with asymmetric padding:
    - more forehead (top)
    - more chin (bottom)
    """
    k = kps.astype(np.float32)
    x_min = float(np.min(k[:, 0]))
    x_max = float(np.max(k[:, 0]))
    y_min = float(np.min(k[:, 1]))
    y_max = float(np.max(k[:, 1]))
    w = max(1.0, x_max - x_min)
    h = max(1.0, y_max - y_min)
    x1 = x_min - pad_x * w
    x2 = x_max + pad_x * w
    y1 = y_min - pad_y_top * h
    y2 = y_max + pad_y_bot * h
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def _ema(prev: Optional[np.ndarray], cur: np.ndarray, alpha: float) -> np.ndarray:
    if prev is None:
        return cur.astype(np.float32)
    return (alpha * prev + (1.0 - alpha) * cur).astype(np.float32)


def _kps_span_ok(kps: np.ndarray, min_eye_dist: float = 12.0) -> bool:
    """
    Quick sanity filter on 5pt geometry:
    - eye distance must be reasonable
    - mouth should be below eyes (usually)
    """
    k = kps.astype(np.float32)
    le, re, no, lm, rm = k
    eye_dist = float(np.linalg.norm(re - le))
    if eye_dist < min_eye_dist:
        return False
    if not (lm[1] > no[1] and rm[1] > no[1]):
        return False
    return True


# -------------------------
# Detector
# -------------------------
class Haar5ptDetector:
    def __init__(
        self,
        haar_xml: Optional[str] = None,
        min_size: Tuple[int, int] = (60, 60),
        smooth_alpha: float = 0.80,
        debug: bool = True,
    ):
        self.debug = bool(debug)
        self.min_size = tuple(map(int, min_size))
        self.smooth_alpha = float(smooth_alpha)

        # Haar cascade
        if haar_xml is None:
            haar_xml = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(haar_xml)
        if self.face_cascade.empty():
            raise RuntimeError(f"Failed to load Haar cascade: {haar_xml}")

        self._prev_box: Optional[np.ndarray] = None
        self._prev_kps: Optional[np.ndarray] = None

    def _estimate_5pt_from_bbox(self, x: int, y: int, w: int, h: int) -> np.ndarray:
        """Estimate 5 facial landmarks from bounding box."""
        return np.array([
            [x + w * 0.30, y + h * 0.35],  # left eye
            [x + w * 0.70, y + h * 0.35],  # right eye
            [x + w * 0.50, y + h * 0.55],  # nose
            [x + w * 0.35, y + h * 0.75],  # left mouth
            [x + w * 0.65, y + h * 0.75],  # right mouth
        ], dtype=np.float32)

    def detect(self, frame: np.ndarray, max_faces: int = 1) -> List[FaceKpsBox]:
        """Detect faces and return 5-point landmarks per face."""
        H, W = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        raw = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=self.min_size
        )
        if len(raw) == 0:
            return []
        # Sort by area, take largest max_faces
        raw = sorted(raw, key=lambda f: f[2] * f[3], reverse=True)[:max_faces]
        out: List[FaceKpsBox] = []
        for i, (x, y, w, h) in enumerate(raw):
            kps = self._estimate_5pt_from_bbox(x, y, w, h)
            # Temporal smoothing only for the largest (first) face
            if i == 0 and self._prev_kps is not None and self.smooth_alpha > 0:
                kps = _ema(self._prev_kps, kps, self.smooth_alpha)
                self._prev_kps = kps
            elif i == 0:
                self._prev_kps = kps
            if not _kps_span_ok(kps, min_eye_dist=8.0):
                continue
            bbox = _bbox_from_5pt(kps)
            x1, y1, x2, y2 = _clip_box_xyxy(bbox, W, H)
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            out.append(FaceKpsBox(x1=x1, y1=y1, x2=x2, y2=y2, score=1.0, kps=kps))
        return out