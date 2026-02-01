#!/usr/bin/env python3
"""
Face Recognition + Face Locking - unified CLI.
"""

import os
os.environ.setdefault("QT_LOGGING_RULES", "*.debug=false;qt.qpa.*=false")

import sys
from pathlib import Path

# Ensure we can import from src
sys.path.insert(0, str(Path(__file__).parent))


def main():
    import argparse
    p = argparse.ArgumentParser(description="Face Recognition & Face Locking")
    p.add_argument("mode", choices=["camera", "enroll", "recognize", "lock"],
                   help="Mode: camera test, enroll, recognize, or lock")
    p.add_argument("--target", default="Gabi", help="Target name for lock mode")
    p.add_argument("--model", default="models/embedder_arcface.onnx")
    args = p.parse_args()

    if args.mode == "camera":
        from src.camera import main as run
        run()
    elif args.mode == "enroll":
        from src.enroll import main as run
        run()
    elif args.mode == "recognize":
        from src.recognize import main as run
        run()
    elif args.mode == "lock":
        from src.lock_app import main as run
        run(target_name=args.target, model_path=args.model)


if __name__ == "__main__":
    main()
