# src/lock_app.py
"""
Face Locking: lock onto a selected identity, track, detect actions, record history.
Run: python -m src.lock_app [target_name]
Example: python -m src.lock_app Gabi
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

from .haar_5pt import Haar5ptDetector, align_face_5pt
from .embed import ArcFaceEmbedderONNX, cosine_similarity
from .recognize import load_db_npz
from .locker import FaceLocker
from .actions import ActionDetector5pt


def main(
    target_name: str = "Gabi",
    model_path: str = "models/embedder_arcface.onnx",
    db_path: Path | str = Path("data/db/face_db.npz"),
    threshold: float = 0.6,
) -> None:
    db_path = Path(db_path)
    if not db_path.exists():
        print(f"[ERROR] Database not found: {db_path}")
        print("Run enrollment first: python -m src.enroll")
        return

    db = load_db_npz(db_path)
    if not db:
        print("[ERROR] Database is empty. Enroll at least one face first.")
        return

    if target_name not in db:
        names = ", ".join(sorted(db.keys()))
        print(f"[ERROR] '{target_name}' not in database. Available: {names}")
        return

    print("=== Face Locking ===")
    print(f"Target: {target_name}")
    print("Keys: q = quit")
    print("-" * 40)

    det = Haar5ptDetector(min_size=(70, 70), smooth_alpha=0.80, debug=False)
    emb = ArcFaceEmbedderONNX(model_path=model_path, input_size=(112, 112), debug=False)
    locker = FaceLocker(target_name=target_name)
    action_det = ActionDetector5pt()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open camera.")
        return

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        vis = frame.copy()
        h, w = vis.shape[:2]

        # 1. Detect faces
        faces = det.detect(frame, max_faces=5)
        match_names: list[str | None] = []

        # 2. Recognize each face
        for f in faces:
            aligned, _ = align_face_5pt(frame, f.kps, out_size=(112, 112))
            vec = emb.embed(aligned).embedding
            best_name = None
            best_sim = -1.0
            for name, ref in db.items():
                sim = cosine_similarity(vec, ref)
                if sim > best_sim and sim >= threshold:
                    best_sim = sim
                    best_name = name
            match_names.append(best_name)

            # Draw box
            color = (0, 255, 0) if best_name else (0, 0, 255)
            cv2.rectangle(vis, (f.x1, f.y1), (f.x2, f.y2), color, 2)
            label = best_name or "?"
            cv2.putText(vis, label, (f.x1, f.y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 3. Update locker
        is_locked, target_idx = locker.update(faces, match_names)

        # 4. When locked: action detection using 5-point landmarks
        if is_locked and target_idx >= 0 and target_idx < len(faces):
            f = faces[target_idx]
            actions = action_det.process_5pt(f.kps)
            for action_type, desc in actions:
                locker.log_action(action_type, desc)

            # Draw locked overlay
            cv2.rectangle(vis, (f.x1, f.y1), (f.x2, f.y2), (0, 255, 255), 3)
            cv2.putText(vis, "LOCKED: " + target_name.upper(),
                        (20, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        else:
            if locker.is_locked:
                cv2.putText(vis, "SEARCHING...", (20, h - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)
            else:
                cv2.putText(vis, f"Looking for: {target_name}",
                            (20, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        cv2.imshow("Face Locking", vis)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"History saved: {locker.history_filename}")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "Gabi"
    main(target_name=target)
