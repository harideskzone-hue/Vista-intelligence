import cv2
import numpy as np
from ultralytics import YOLO
from edge.boundary.adapter import BoundaryTrackingAdapter

def test_native_profile():
    print("--- Simulating Native Cadence ---")
    adapter = BoundaryTrackingAdapter()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for i in range(4):
        adapter.detect_and_track(frame, timestamp=i * 0.033)
    tracker = adapter.model.predictor.trackers[0]
    print(f"Native Profile -> max_frames_lost: {tracker.max_frames_lost} | match_thresh: {tracker.args.match_thresh}")
    assert tracker.max_frames_lost == 30, f"Expected 30, got {tracker.max_frames_lost}"
    assert tracker.args.match_thresh == 0.8, f"Expected 0.8, got {tracker.args.match_thresh}"

def test_sparse_profile():
    print("--- Simulating Sparse Cadence ---")
    adapter = BoundaryTrackingAdapter()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for i in range(4):
        adapter.detect_and_track(frame, timestamp=i * 0.833)
    tracker = adapter.model.predictor.trackers[0]
    print(f"Sparse Profile -> max_frames_lost: {tracker.max_frames_lost} | match_thresh: {tracker.args.match_thresh}")
    assert tracker.max_frames_lost == 10, f"Expected 10, got {tracker.max_frames_lost}"
    assert tracker.args.match_thresh == 0.9, f"Expected 0.9, got {tracker.args.match_thresh}"

if __name__ == "__main__":
    print("Testing dynamic tracker lifecycle...")
    test_native_profile()
    test_sparse_profile()
    print("SUCCESS: Dynamic mutation verified!")
