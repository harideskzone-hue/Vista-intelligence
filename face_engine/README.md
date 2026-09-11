# Best Photo Picker 📸

Best Photo Picker is an open-source, AI-powered system designed to analyze and score photos or live camera feeds automatically. By leveraging **YOLO** (for person and face detection) and **MediaPipe** (for dense facial and body landmarks), it assesses images for factors like eye openness, gaze, head pose, body orientation, and even smiles, ultimately outputting an objective quality score.

## Features ✨

*   **Intelligent Photo Scoring**: Evaluates photos on multiple metrics (Face Group, Body Group, Preflight blur checks, Frame centering) and produces a final score (0–100).
*   **Live Camera Calibration & Capture**: using `live_scorer.py`, you can get real-time overlay visualization on top of video. It automatically captures and saves "best poses," rejecting those with motion blur or bad posture.
*   **Web Dashboard UI**: `app.py` spins up a Flask API backend with a frontend web dashboard to upload batches of photos and instantly see their scores, ranks, and detected errors.
*   **Command Line Processing**: Process directories or individual images headlessly.

## Tech Stack & Architecture 🏗️

The application follows a strictly structured detection pipeline:
1.  **Person Detection**: Ultralytics YOLO26n-person on the full image.
2.  **Face Detection**: Ultralytics YOLO26n-face on the cropped person bounding box.
3.  **Face Landmarking**: MediaPipe FaceLandmarker extracting 478 points on the face crop.
4.  **Body Landmarking**: MediaPipe PoseLandmarker extracting 33 points on the full image.
5.  **Aggregator Pipeline**: Combines independent module scores (like `smile`, `eye_openness`, `posture`) using a configurable weighted average list.

## Installation 🛠️

1. **Clone the repository and enter the directory.**
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Download Required Models**:
   Create a `models/` directory inside your project folder and fetch the `yolo26n` and `mediapipe` models according to the specs. A download helper (`models/download_models.sh`) might be available:
   - `models/yolo26n-person.pt`
   - `models/yolo26n-face.pt`
   - `models/face_landmarker.task`
   - `models/pose_landmarker_full.task`

## Usage 🚀

### 1. Web Application (UI)
Start the Flask backend to access the browser-based Best Photo Picker interface.
```bash
python app.py
```
*   **Access**: Open `http://localhost:5050` in your web browser.
*   **Action**: Drop images into the UI to score and rank them automatically.

### 2. Live Camera Picker
Great for taking photos natively. It connects to your camera feed (you can configure the `camera_source` inside `live_scorer.py`) and captures photos only when your pose is stable, clear, and high-scoring.
```bash
python live_scorer.py
```

### 3. Command Line Scoring
Score single photos or whole directories natively from the terminal.
```bash
python score_folder.py --input ./photos --output ./results
```

## Configuration ⚙️
Settings like camera FOV, minimum body orientation, bounds for face constraints, weights for face vs. body groups, and edge-cutoff thresholds can be tweaked inside `pose_scorer/config.py`. 
