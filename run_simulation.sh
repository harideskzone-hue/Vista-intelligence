#!/bin/bash
set -e

# Usage: bash run_simulation.sh test_videos/my_video.mp4

VIDEO_PATH="$1"

if [ -z "$VIDEO_PATH" ]; then
    echo "Usage: bash run_simulation.sh <path_to_video.mp4>"
    exit 1
fi

if [ ! -f "$VIDEO_PATH" ]; then
    echo "Error: Video file '$VIDEO_PATH' not found."
    exit 1
fi

# Convert to absolute path
VIDEO_ABS_PATH="$(cd "$(dirname "$VIDEO_PATH")" && pwd)/$(basename "$VIDEO_PATH")"

echo "Setting cameras.json to use video: $VIDEO_ABS_PATH"

cat > cameras.json << EOL
[
  {
    "id": "cam_video_sim",
    "source": "$VIDEO_ABS_PATH",
    "label": "Video Simulation",
    "enabled": true
  }
]
EOL

echo "Starting Face API in the background..."
KMP_DUPLICATE_LIB_OK=True PYTHONPATH=./face_api:. python3 face_api/run.py &
API_PID=$!

# Wait for API to be ready
echo "Waiting for Face API to initialize..."
for i in {1..30}; do
    if curl -sf "http://localhost:5001/api/health" > /dev/null 2>&1; then
        echo "Face API is ready!"
        break
    fi
    sleep 1
done

echo "Starting Live Scorer..."
# Run the pipeline
KMP_DUPLICATE_LIB_OK=True PYTHONPATH=./face_api:. python3 face_engine/live_scorer.py

# Cleanup when live scorer stops
echo "Live Scorer stopped. Shutting down Face API..."
kill $API_PID
echo "Simulation complete."
