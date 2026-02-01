# src/recognize.py
"""
Multi-face recognition (CPU-friendly) using your now-stable pipeline:
Haar (multi-face) -> FaceMesh 5pt (per-face ROI) -> align_face_5pt (112x112)
-> ArcFace ONNX embedding -> cosine distance to DB -> label each face.
Run:
python -m src.recognize
Keys:
q : quit
r : reload DB from disk (data/db/face_db.npz)
+/- : adjust threshold (distance) live
d : toggle debug overlay
Notes:
- We run FaceMesh on EACH Haar face ROI (not the full frame). This avoids the
“FaceMesh points not consistent with Haar box” problem and enables multi-face.
- DB is expected from enroll: data/db/face_db.npz (name -> embedding vector)
- Distance definition: cosine_distance = 1 - cosine_similarity.
Since embeddings are L2-normalized, cosine_similarity = dot(a,b).
"""

from __future__ import annotations
import time
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from .haar_5pt import align_face_5pt
from .embed import ArcFaceEmbedderONNX


# -------------------------
# Data
# -------------------------
@dataclass
class FaceDet:
    x1: int
    y1: int
    x2: int
    y2: int
    score: float
    kps: np.ndarray  # (5,2) float32 in FULL-frame coords


@dataclass
class MatchResult:
    name: Optional[str]
    distance: float
    similarity: float
    accepted: bool


# -------------------------
# Math helpers
# -------------------------
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float32)
    b = b.reshape(-1).astype(np.float32)
    return float(np.dot(a, b))


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return 1.0 - cosine_similarity(a, b)


def _clip_xyxy(x1: float, y1: float, x2: float, y2: float, W: int, H: int) -> Tuple[int, int, int, int]:
    x1 = int(max(0, min(W - 1, round(x1))))
    y1 = int(max(0, min(H - 1, round(y1))))
    x2 = int(max(0, min(W - 1, round(x2))))
    y2 = int(max(0, min(H - 1, round(y2))))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def _bbox_from_5pt(
    kps: np.ndarray,
    pad_x: float = 0.55,
    pad_y_top: float = 0.85,
    pad_y_bot: float = 1.15,
) -> np.ndarray:
    """
    Build a nicer face-like bbox from 5 points with asymmetric padding.
    kps: (5,2) in full-frame coords
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


def _kps_span_ok(kps: np.ndarray, min_eye_dist: float) -> bool:
    """
    Minimal geometry sanity:
    - eyes not collapsed
    - mouth generally below nose
    """
    k = kps.astype(np.float32)
    le, re, no, lm, rm = k
    eye_dist = float(np.linalg.norm(re - le))
    if eye_dist < float(min_eye_dist):
        return False
    if not (lm[1] > no[1] and rm[1] > no[1]):
        return False
    return True


# -------------------------
# DB helpers
# -------------------------
def load_db_npz(db_path: Path) -> Dict[str, np.ndarray]:
    if not db_path.exists():
        return {}
    data = np.load(str(db_path), allow_pickle=True)
    out: Dict[str, np.ndarray] = {}
    for k in data.files:
        out[k] = np.asarray(data[k], dtype=np.float32).reshape(-1)
    return out


# -------------------------
# Main recognition loop
# -------------------------
def main() -> None:
    from .haar_5pt import Haar5ptDetector

    db_path = Path("data/db/face_db.npz")
    threshold_dist = 0.5  # cosine_distance <= 0.5 means same person

    db = load_db_npz(db_path)
    det = Haar5ptDetector(min_size=(70, 70), smooth_alpha=0.80, debug=False)
    emb = ArcFaceEmbedderONNX(model_path="models/embedder_arcface.onnx", input_size=(112, 112))

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera.")

    print("Recognize: q=quit, r=reload DB, +/-=threshold")
    t0 = time.time()
    frames = 0
    fps = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        vis = frame.copy()
        faces = det.detect(frame, max_faces=5)

        for f in faces:
            aligned, _ = align_face_5pt(frame, f.kps, out_size=(112, 112))
            vec = emb.embed(aligned).embedding
            best_name = None
            best_dist = 1.0
            for name, ref in db.items():
                d = cosine_distance(vec, ref)
                if d < best_dist and d <= threshold_dist:
                    best_dist = d
                    best_name = name

            color = (0, 255, 0) if best_name else (0, 0, 255)
            cv2.rectangle(vis, (f.x1, f.y1), (f.x2, f.y2), color, 2)
            label = best_name or "Unknown"
            cv2.putText(vis, label, (f.x1, f.y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        frames += 1
        if time.time() - t0 >= 1.0:
            fps = frames / (time.time() - t0)
            frames = 0
            t0 = time.time()
        cv2.putText(vis, f"FPS: {fps:.1f}  dist<={threshold_dist:.2f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Face Recognition", vis)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if k == ord("r"):
            db = load_db_npz(db_path)
            print("DB reloaded")
        if k in (ord("+"), ord("=")):
            threshold_dist = min(1.0, threshold_dist + 0.05)
        if k == ord("-"):
            threshold_dist = max(0.1, threshold_dist - 0.05)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()