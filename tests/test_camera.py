import numpy as np
import os
import sys
import time
import cv2
import pytest
from unittest.mock import patch, MagicMock

# MOCK ULTRALYTICS TO PREVENT TORCH OPENMP COLLISION WITH FAISS DURING LIVE_SCORER IMPORT
_orig_ultralytics = sys.modules.get('ultralytics')
sys.modules['ultralytics'] = MagicMock()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from face_engine.live_scorer import CameraStream, CameraProcessor, FaceDetectorThread

# Restore real ultralytics so downstream detector/tracker tests are not contaminated
if _orig_ultralytics is not None:
    sys.modules['ultralytics'] = _orig_ultralytics
else:
    sys.modules.pop('ultralytics', None)

class DummyModel:
    def __call__(self, frame, conf, verbose):
        mock_res = MagicMock()
        mock_res.boxes = []
        return [mock_res]

def test_camera_stream_no_fallback():
    # Attempt to open an invalid RTSP url
    stream = CameraStream("rtsp://invalid_url_that_does_not_exist/test")
    assert stream.stopped is True
    assert stream.stream.isOpened() is False

def test_camera_stream_success(tmp_path):
    # Create a dummy valid video file
    video_path = str(tmp_path / "test.mp4")
    # OpenCV might fail to write with some fourccs if not installed, using uncompressed or widely supported
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (100, 100))
    for _ in range(10):
        out.write(np.zeros((100, 100, 3), dtype=np.uint8))
    out.release()
    
    stream = CameraStream(video_path)
    assert stream.stopped is False
    assert stream.stream.isOpened() is True
    stream.start()
    time.sleep(0.5)
    ret, frame, frame_id = stream.read()
    assert ret is True
    assert frame is not None
    stream.stop()

def test_camera_processor_independent(tmp_path):
    # Two processors, one fails, one succeeds
    video_path = str(tmp_path / "test.mp4")
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (100, 100))
    for _ in range(10):
        out.write(np.zeros((100, 100, 3), dtype=np.uint8))
    out.release()

    # Camera 1 succeeds
    proc1 = CameraProcessor("cam1", video_path, DummyModel())
    assert proc1.stream.stopped is False
    
    # Camera 2 fails
    proc2 = CameraProcessor("cam2", "rtsp://invalid", DummyModel())
    assert proc2.stream.stopped is True
    
    proc1.stop()
    if not proc2.stream.stopped:
        proc2.stop()
