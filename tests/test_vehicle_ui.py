import pytest
import asyncio
from fastapi.testclient import TestClient
from face_api.run import app
from edge.events.event_hub import get_event_hub
from edge.vehicle.manager import get_vehicle_manager

client = TestClient(app)

@pytest.fixture(autouse=True)
def setup_teardown():
    hub = get_event_hub()
    with hub._lock:
        with hub._get_conn() as conn:
            conn.execute("DELETE FROM system_events")
    
    manager = get_vehicle_manager()
    manager._watchlist.clear()
    manager._emitted_detection.clear()
    
    yield

def test_vehicle_route_renders():
    response = client.get("/vehicle")
    assert response.status_code == 200
    assert "Live Vehicle Feed" in response.text
    assert "ANPR Query" in response.text

def test_navigation_dock_links():
    response = client.get("/boundary")
    assert response.status_code == 200
    assert 'href="/vehicle"' in response.text

def test_api_contract_watchlist_crud():
    response = client.post("/api/vehicle/watchlist", json={
        "plate_text": "UI1234",
        "reason": "Stolen",
        "severity": "CRITICAL"
    })
    assert response.status_code == 200
    assert response.json()["status"] == "success"

    response = client.get("/api/vehicle/watchlist")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["plate_text"] == "UI1234"

    response = client.delete("/api/vehicle/watchlist/UI1234")
    assert response.status_code == 200

    response = client.get("/api/vehicle/watchlist")
    assert len(response.json()) == 0

def test_api_contract_event_query():
    hub = get_event_hub()
    from edge.events.event_hub import SystemEvent
    import time
    import uuid
    
    hub.log_event(
        subsystem="VEHICLE",
        camera_id="CAM_1",
        event_type="VEHICLE_DETECTED",
        timestamp=time.time(),
        details="Detected",
        severity="INFO",
        metadata={"plate_text": "DEF567", "vehicle_class": "car"}
    )
    
    hub.log_event(
        subsystem="VEHICLE",
        camera_id="CAM_2",
        event_type="VEHICLE_IDENTIFIED",
        timestamp=time.time(),
        details="Identified",
        severity="INFO",
        metadata={"plate_text": "ABC123", "vehicle_class": "truck"}
    )

    res = client.get("/api/vehicle/events")
    assert res.status_code == 200
    assert len(res.json()) == 2
    
    res = client.get("/api/vehicle/events?camera_id=CAM_2")
    assert len(res.json()) == 1
    assert res.json()[0]["metadata"]["plate_text"] == "ABC123"

    res = client.get("/api/vehicle/events?search_query=DEF")
    assert len(res.json()) == 1
    assert res.json()[0]["metadata"]["plate_text"] == "DEF567"

def test_security_clip_path_not_leaked():
    hub = get_event_hub()
    from edge.events.event_hub import SystemEvent
    import time
    import uuid
    
    hub.log_event(
        subsystem="VEHICLE",
        camera_id="CAM_1",
        event_type="VEHICLE_DETECTED",
        timestamp=time.time(),
        details="Detected",
        severity="INFO",
        metadata={"plate_text": "SECURE", "clip_path": "/absolute/path/to/clip.mp4"}
    )

    res = client.get("/api/vehicle/events")
    events = res.json()
    assert len(events) == 1
    assert "clip_path" not in events[0] or events[0].get("clip_path") is None or not str(events[0].get("clip_path")).startswith("/")

def test_negative_path_no_vehicles():
    import os
    video_path = os.path.join(os.path.dirname(__file__), "../tests/data/no_vehicles.mp4")
    if os.path.exists(video_path):
        with open(video_path, "rb") as f:
            res = client.post(
                "/api/vehicle/test-video",
                files={"file": ("no_vehicles.mp4", f, "video/mp4")},
                data={"camera_id": "TEST_CAM"}
            )
        assert res.status_code == 200
        data = res.json()
        assert data["success"] is True
        assert data["vehicles_detected"] == 0
        assert data["clips_generated"] == 0
