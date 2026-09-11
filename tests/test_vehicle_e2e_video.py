import pytest
import os
import sys
import time
import subprocess
from fastapi.testclient import TestClient
from pathlib import Path

# Setup paths
sys.path.insert(0, os.path.abspath('.'))
from face_api.run import app
from edge.vehicle.manager import get_vehicle_manager

client = TestClient(app)

@pytest.fixture(scope="module")
def generate_negative_video():
    path = "/tmp/no_vehicles.mp4"
    subprocess.run(["ffmpeg", "-f", "lavfi", "-i", "color=c=black:s=640x480:d=1", "-vcodec", "libx264", "-y", path], capture_output=True)
    yield path
    if os.path.exists(path):
        os.remove(path)

@pytest.fixture(autouse=True)
def reset_manager():
    # Force fresh state for each test
    manager = get_vehicle_manager()
    manager._adapters.clear()
    manager._watchlist.clear()
    manager._emitted_detection.clear()
    manager._emitted_identification.clear()
    manager._emitted_match.clear()

def test_real_video_e2e_positive_path():
    video_path = "/Users/hariharans/Documents/SIH26187/Vehicle Intelligence/SIH26187/input/a11_benchmark/highway_fast.mp4"
    if not os.path.exists(video_path):
        pytest.skip(f"Test video {video_path} not found")

    with open(video_path, "rb") as f:
        response = client.post(
            "/api/vehicle/test-video",
            files={"file": ("highway_fast.mp4", f, "video/mp4")},
            data={"camera_id": "CAM_E2E_POS"}
        )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["vehicles_detected"] > 0
    assert len(data["vehicles"]) > 0
    assert data["total_frames"] > 0
    
    for veh in data["vehicles"]:
        assert veh["make"] == "UNKNOWN", "Make must not be fabricated"
        assert veh["model"] == "UNKNOWN", "Model must not be fabricated"

    events_response = client.get("/api/vehicle/events")
    assert events_response.status_code == 200
    events = events_response.json()
    
    pos_events = [e for e in events if e["camera_id"] == "CAM_E2E_POS"]
    assert len(pos_events) > 0
    
    for e in pos_events:
        if "clip_path" in e:
            assert not e["clip_path"].startswith("/")
            assert not str(e["clip_path"]).startswith("C:\\")
            
    identified_events = [e for e in pos_events if e["event_type"] == "VEHICLE_IDENTIFIED"]
    if identified_events:
        time.sleep(1.0)
        e_id = identified_events[0]["id"]
        clip_response = client.get(f"/api/events/{e_id}/clip")
        assert clip_response.status_code in (200, 206, 404)

def test_real_video_e2e_negative_path(generate_negative_video):
    video_path = generate_negative_video
    
    client.post("/api/vehicle/watchlist", json={"plate_text": "FAK3PL8", "severity": "CRITICAL"})

    with open(video_path, "rb") as f:
        response = client.post(
            "/api/vehicle/test-video",
            files={"file": ("no_vehicles.mp4", f, "video/mp4")},
            data={"camera_id": "CAM_E2E_NEG"}
        )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["vehicles_detected"] == 0
    assert len(data["vehicles"]) == 0
    
    events_response = client.get("/api/vehicle/events")
    assert events_response.status_code == 200
    events = events_response.json()
    
    neg_events = [e for e in events if e["camera_id"] == "CAM_E2E_NEG"]
    assert len(neg_events) == 0

def test_camera_state_isolation_e2e():
    manager = get_vehicle_manager()
    a1 = manager.get_or_create_adapter("cam_1")
    a2 = manager.get_or_create_adapter("cam_2")
    
    assert a1 is not a2
    assert "cam_1" in manager._adapters
    assert "cam_2" in manager._adapters
    assert id(a1.pipeline) != id(a2.pipeline)
    assert id(a1.pipeline.tracker) != id(a2.pipeline.tracker)
