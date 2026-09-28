#!/bin/bash
echo "Starting VISTA AI on Hugging Face Spaces..."

# Define where models/data will live in Hugging Face (persistent storage if configured, otherwise ephemeral /app/data)
export SIH26187_DATA=/app/data
export PYTHONPATH=/app/face_api:/app
export SIH26187_VEHICLE_INT=/app/Vehicle\ Intelligence/SIH26187

mkdir -p $SIH26187_DATA
mkdir -p /app/logs

# Start the FastAPI Backend
echo "Starting FastAPI Backend..."
python3 /app/face_api/run.py &
API_PID=$!

# Wait for API to be healthy
echo "Waiting for API to start on port $PORT..."
sleep 5

# Start the AI Engine (Live Scorer)
echo "Starting AI Engine..."
python3 -u /app/face_engine/live_scorer.py &
SCORER_PID=$!

# Keep container alive by waiting on processes
wait -n $API_PID $SCORER_PID
