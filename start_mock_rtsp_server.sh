#!/bin/bash
# start_mock_rtsp_server.sh
# Starts MediaMTX and loops test_video.mp4 to 12 RTSP endpoints

echo "Starting mediamtx..."
./mediamtx > mediamtx.log 2>&1 &
MEDIAMTX_PID=$!
sleep 2

echo "Pushing video streams..."
for i in {1..12}; do
    ffmpeg -re -stream_loop -1 -i ./test_video.mp4 -c copy -f rtsp rtsp://localhost:8554/cam$i > /dev/null 2>&1 &
    echo "Stream cam$i started."
done

echo "RTSP server running. Press Ctrl+C to stop."
wait $MEDIAMTX_PID
