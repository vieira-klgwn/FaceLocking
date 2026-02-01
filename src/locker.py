# src/locker.py
"""
Face locking state and history recording.
Tracks a selected identity, tolerates brief recognition failures, records actions.
"""

import time
from pathlib import Path
from typing import List, Optional, Any


class FaceLocker:
    """
    Manages locking onto a specific identity.
    Uses spatial overlap (IoU) to maintain lock when recognition briefly fails.
    """

    def __init__(self, target_name: str, history_dir: str = "data/history"):
        self.target_name = target_name
        self.history_dir = Path(history_dir)
        self.history_dir.mkdir(parents=True, exist_ok=True)

        self.is_locked = False
        self.last_seen_time = 0.0
        self.lock_timeout = 2.0  # Seconds before releasing if face disappears
        self.last_bbox: Optional[List[float]] = None  # [x1, y1, x2, y2]

        timestamp = time.strftime("%Y%m%d%H%M%S")
        self.history_filename = self.history_dir / f"{target_name.lower()}_history_{timestamp}.txt"
        self._init_history_file()

    def _init_history_file(self) -> None:
        with open(self.history_filename, "w") as f:
            f.write(f"Face Locking History: {self.target_name}\n")
            f.write(f"Started at: {time.ctime()}\n")
            f.write("-" * 40 + "\n")
            f.write(f"{'Timestamp':<25} | {'Action':<12} | {'Description'}\n")
            f.write("-" * 40 + "\n")
        print(f"[LOCKER] History: {self.history_filename}")

    def log_action(self, action_type: str, description: str) -> None:
        """Record an action to the history file."""
        if not self.is_locked:
            return
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(self.history_filename, "a") as f:
            f.write(f"{ts:<25} | {action_type:<12} | {description}\n")

    def update(
        self,
        faces: List[Any],
        match_names: List[Optional[str]],
    ) -> tuple[bool, int]:
        """
        Update lock state from detected faces and recognition results.
        Returns (is_locked, target_face_index) or (False, -1).
        """
        now = time.time()

        # 1. Find target in recognition results
        target_idx = -1
        for i, name in enumerate(match_names):
            if name and name.lower() == self.target_name.lower():
                target_idx = i
                break

        if target_idx >= 0 and target_idx < len(faces):
            self.is_locked = True
            self.last_seen_time = now
            f = faces[target_idx]
            self.last_bbox = [float(f.x1), float(f.y1), float(f.x2), float(f.y2)]
            return True, target_idx

        # 2. Already locked but target not recognized this frame: try spatial tracking
        if self.is_locked:
            if now - self.last_seen_time > self.lock_timeout:
                print(f"[LOCKER] Lock released for {self.target_name} (timeout)")
                self.is_locked = False
                self.last_bbox = None
                return False, -1

            if self.last_bbox and faces:
                best_iou = 0.0
                best_idx = -1
                for i, f in enumerate(faces):
                    box = [float(f.x1), float(f.y1), float(f.x2), float(f.y2)]
                    iou = self._iou(self.last_bbox, box)
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = i
                if best_iou > 0.3:
                    f = faces[best_idx]
                    self.last_bbox = [float(f.x1), float(f.y1), float(f.x2), float(f.y2)]
                    self.last_seen_time = now
                    return True, best_idx

        return False, -1

    @staticmethod
    def _iou(a: List[float], b: List[float]) -> float:
        """Intersection over Union of two boxes [x1,y1,x2,y2]."""
        xA = max(a[0], b[0])
        yA = max(a[1], b[1])
        xB = min(a[2], b[2])
        yB = min(a[3], b[3])
        inter = max(0, xB - xA) * max(0, yB - yA)
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        return inter / (area_a + area_b - inter + 1e-6)
