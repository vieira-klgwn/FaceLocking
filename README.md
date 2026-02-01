


# Face Recognition with ArcFace ONNX and 5-Point Alignment

This project implements a **modular face recognition system** using **ArcFace ONNX embeddings** and **5-point facial alignment**. It is designed to run efficiently on **CPU**, and each stage (detection, alignment, embedding, enrollment, recognition) is testable and independent.  

---

## Table of Contents

- [Requirements](#requirements)  
- [Setup](#setup)  
- [Running the Project](#running-the-project)  
- [Face Enrollment](#face-enrollment)  
- [Threshold for Recognition](#threshold-for-recognition)  
- [Project Structure](#project-structure)  

---

## Requirements

- Python 3.9+  
- Webcam  
- OS: macOS, Linux, or Windows  

**Dependencies:**

```bash
pip install opencv-python numpy onnxruntime scipy tqdm mediapipe
````

| Package                | Purpose                                              |
| ---------------------- | ---------------------------------------------------- |
| OpenCV (opencv-python) | Camera access, Haar face detection, image processing |
| NumPy                  | Numerical operations                                 |
| ONNX Runtime           | CPU inference for ArcFace embeddings                 |
| SciPy                  | Distance computation and evaluation                  |
| MediaPipe              | 5-point facial landmark extraction                   |
| tqdm                   | Progress bars for enrollment and evaluation          |

---

## Setup

1. Create a Python virtual environment:

```
python -m venv .venv
```

2. Activate the environment:

* macOS / Linux:

```
source .venv/bin/activate
```

* Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

3. Install dependencies:

```bash
pip install --upgrade pip
pip install opencv-python numpy onnxruntime scipy tqdm mediapipe
```

4. Verify webcam access:

```bash
python -m src.camera
```

Expected result: a live camera window opens, motion appears smooth, FPS is displayed, pressing `q` exits cleanly.

---

## Running the Project

1. **Face Detection & Landmark Visualization:**

```bash
python -m src.haar_5pt
```

* Shows bounding boxes around detected faces with 5-point landmarks (eyes, nose, mouth corners).

2. **Face Enrollment:**

```bash
python -m src.enroll
```

* Captures faces from the webcam.
* Aligns each face using 5-point landmarks.
* Extracts embeddings with ArcFace ONNX.
* Stores embeddings in `data/db/face_db.npz` and metadata in `data/db/face_db.json`.

3. **Live Recognition:**

```bash
python -m src.recognize
```

* Detects and aligns faces in real-time.
* Extracts embeddings.
* Compares query embeddings to stored embeddings using cosine similarity.
* Returns the recognized identity if the distance is below the defined threshold.

4. **Face Locking:**

```bash
python main.py lock --target <Name>
# or: python -m src.lock_app <Name>
```

* Locks onto the specified enrolled identity.
* Tracks the face as it moves; tolerates brief recognition failures.
* Detects actions: face moved left/right, eye blink, smile/laugh.
* Records actions to `data/history/<name>_history_<timestamp>.txt`.

---

## Face Locking (Assignment)

* **Manual selection**: Choose one enrolled identity to lock (e.g. `Gabi`).
* **Locking**: When the face is recognized, the system locks onto it and displays "LOCKED".
* **Stable tracking**: Uses spatial overlap (IoU) to maintain lock when recognition briefly fails.
* **Action detection**: face moved left/right, smile (blink requires MediaPipe Face Mesh).
* **History file**: Format `<face>_history_<timestamp>.txt` with timestamp, action type, description.

---

## Face Enrollment

* Each enrolled identity is stored as **one or more embeddings** in the database.
* Images are aligned to **112×112** using the 5-point landmarks.
* Stored embeddings are **L2-normalized** for reliable similarity comparison.
* Enrollment is necessary before live recognition can correctly identify faces.

---

## Threshold for Recognition

* Recognition is performed using **cosine similarity or L2 distance** between embeddings.
* A **distance threshold** determines if two faces belong to the same person:

```text
Recommended default threshold: 0.8 (L2 distance)
```

* Faces with distance **≤ threshold** are considered the same person.
* Faces with distance **> threshold** are considered different (open-set recognition).

---

## Project Structure

```
face-recognition-5pt/
├── data/
│   ├── enroll/         # Enrollment images
│   ├── db/             # Face database (embeddings + metadata)
│   └── history/        # Face locking action history files
├── models/
│   └── embedder_arcface.onnx
├── src/
│   ├── actions.py       # Action detection (blink, smile, move)
│   ├── align.py         # Face alignment
│   ├── camera.py        # Webcam test
│   ├── detect.py        # Face detection test
│   ├── embed.py         # Embedding extraction
│   ├── enroll.py        # Enrollment pipeline
│   ├── evaluate.py      # Threshold tuning
│   ├── haar_5pt.py      # Haar + 5-point landmarks
│   ├── landmarks.py     # Landmark tests
│   ├── landmarks_mp.py  # MediaPipe Face Mesh (for actions)
│   ├── locker.py        # Face locking state & history
│   ├── lock_app.py      # Face locking application
│   └── recognize.py     # Live recognition pipeline
├── init_project.py       # Automated project setup
└── README.md
```

* **Modular design** allows replacing models or modifying stages without rewriting the entire pipeline.

---
