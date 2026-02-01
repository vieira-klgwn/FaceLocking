# How to Test Face Locking (Assignment)

> **Note**: You may see Qt or font warnings when running. These are harmless; the app works.

## 1. Setup

```bash
cd face-recognition-5pt
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python init_project.py
```

## 2. Download Model (if needed)

```bash
python download_model.py
```

If the model file is empty or missing, this downloads the ArcFace ONNX model.

## 3. Enroll a Face

```bash
python main.py enroll
# or: python -m src.enroll
```

- Enter your name (e.g. "Gabi" or "Fani")
- Press SPACE to capture samples, or 'a' for auto-capture
- Collect at least 5-15 samples
- Press 's' to save enrollment
- Press 'q' to quit

## 4. Run Face Locking

```bash
python main.py lock --target Gabi
# or: python -m src.lock_app Gabi
```

Replace `Gabi` with the name you enrolled.

### What to Test

1. **Lock**: Look at the camera. When recognized, you should see "LOCKED: GABI" and a yellow box.
2. **Tracking**: Move your face left/right. The system should stay locked.
3. **Actions**: Move left/right - actions should be logged. Check `data/history/<name>_history_<timestamp>.txt`.
4. **Stability**: Briefly look away - lock should hold for ~2 seconds before releasing.
5. **Multi-face**: If someone else appears, the system should stay focused on the target.

### History File

Location: `data/history/<name>_history_YYYYMMDDHHMMSS.txt`

Format:
```
Timestamp                  | Action       | Description
2026-02-01 22:15:30       | move         | face moved left
```

## 5. Run Recognition (without locking)

```bash
python main.py recognize
# or: python -m src.recognize
```

- Press +/- to adjust threshold
- Press 'r' to reload database
- Press 'q' to quit
