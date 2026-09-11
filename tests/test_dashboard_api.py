import pytest
import os
import sys
import json
from unittest.mock import patch, MagicMock

# Mock insightface and bcrypt
sys.modules['insightface'] = MagicMock()
sys.modules['insightface.app'] = MagicMock()
sys.modules['bcrypt'] = MagicMock()

from fastapi.testclient import TestClient

@pytest.fixture
def temp_cameras_json(tmp_path):
    cam_file = tmp_path / "cameras.json"
    cam_file.write_text("[]")
    # Patch routes module variable
    import app.api.routes as routes
    routes._CAMERAS_JSON = str(cam_file)
    return str(cam_file)

@pytest.fixture
def mock_dependencies(tmp_path):
    os.environ["SIH26187_DATA"] = str(tmp_path)
    # Clear globals in face_service
    import app.services.face_service as fs
    fs._identity_manager = None
    fs._watchlist = None
    fs._temporal_matcher = None
    fs._face_matcher = None
    fs._best_frame_selector = None
    fs._duplicate_filter = None
    
    # We clear the live events in routes
    import app.api.routes as routes
    routes._live_events.clear()

    with patch("app.services.face_service.get_storage"), \
         patch("app.services.face_service.get_vector_db"), \
         patch("app.services.face_service.get_encoder"), \
         patch("app.api.routes._validate_source", return_value=(True, "")):
        yield

@pytest.fixture
def client(temp_cameras_json, mock_dependencies):
    from face_api.run import create_app
    app = create_app()
    # disable auth requirement for tests
    app.dependency_overrides = {}
    
    # Actually create_app adds auth middleware, so to bypass auth we mock is_session_valid
    with patch("face_api.run.is_session_valid", return_value=True):
        yield TestClient(app)

def test_camera_add(client):
    res = client.post("/api/cameras", json={"source": "0", "label": "USB Cam", "enabled": True})
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert "cam_" in data["camera"]["id"]
    assert data["camera"]["source"] == "0"

def test_camera_remove(client):
    # Add
    res = client.post("/api/cameras", json={"source": "0", "label": "USB Cam", "enabled": True})
    cam_id = res.json()["camera"]["id"]
    
    # Delete
    res_del = client.delete(f"/api/cameras/{cam_id}")
    assert res_del.status_code == 200
    assert res_del.json()["success"] is True
    
    # List
    res_list = client.get("/api/cameras")
    assert len(res_list.json()["cameras"]) == 0

def test_camera_enable_disable(client):
    res = client.post("/api/cameras", json={"source": "0", "label": "USB Cam", "enabled": True})
    cam_id = res.json()["camera"]["id"]
    
    res_update = client.put(f"/api/cameras/{cam_id}", json={"enabled": False})
    assert res_update.status_code == 200
    assert res_update.json()["camera"]["enabled"] is False

def test_multiple_camera_listing(client):
    client.post("/api/cameras", json={"source": "0", "label": "Cam1", "enabled": True})
    client.post("/api/cameras", json={"source": "1", "label": "Cam2", "enabled": True})
    res = client.get("/api/cameras")
    assert res.status_code == 200
    assert len(res.json()["cameras"]) == 2

def test_camera_status(client):
    res = client.get("/api/camera/status")
    assert res.status_code == 200
    assert "active" in res.json()

def test_rtsp_credentials_not_exposed(client):
    res = client.post("/api/cameras", json={
        "source": "rtsp://admin:secret123@192.168.1.100/stream", 
        "label": "RTSP Cam", 
        "enabled": True
    })
    # Response from POST should be masked
    assert res.status_code == 200
    cam_data = res.json()["camera"]
    assert cam_data["source"] == "rtsp://***:***@192.168.1.100/stream"
    
    # List cameras should be masked
    res_list = client.get("/api/cameras")
    assert res_list.json()["cameras"][0]["source"] == "rtsp://***:***@192.168.1.100/stream"

def test_enrollment_endpoint(client, tmp_path):
    with patch("app.services.face_service.FaceService.enroll_person_bulk") as mock_enroll:
        mock_enroll.return_value = {'success': True, 'person_id': 'person_xyz'}
        res = client.post("/api/enroll", data={"name": "Test", "role": "STAFF"}, files={"image": ("test.jpg", b"dummy")})
        assert res.status_code == 200
        assert res.json()["person_id"] == "person_xyz"

def test_unknown_event_display(client):
    with patch("app.services.face_service.FaceService.recognize_live_frame") as mock_rec:
        mock_rec.return_value = {
            "success": True,
            "match_status": "UNKNOWN",
            "matched_identity_id": None,
            "action": "none"
        }
        
        # We need multipart form data
        res = client.post("/api/recognize_live", files={"image": ("test.jpg", b"dummy_bytes")}, data={"camera_id": "CAM-1", "track_id": "T-1"})
        print("Recognize Live Result:", res.json())
        assert res.status_code == 200
        
        res_ev = client.get("/api/events/live")
        events = res_ev.json()["events"]
        assert len(events) == 1
        assert events[0]["match_status"] == "UNKNOWN"

def test_match_event_display(client):
    with patch("app.services.face_service.FaceService.recognize_live_frame") as mock_rec:
        mock_rec.return_value = {
            "success": True,
            "match_status": "MATCH",
            "temporal_state": "CONFIRMED",
            "matched_identity_id": "person_123",
            "action": "saved_evidence"
        }
        
        # We need multipart form data
        res = client.post("/api/recognize_live", files={"image": ("test.jpg", b"dummy_bytes")}, data={"camera_id": "CAM-1", "track_id": "T-1"})
        print("Recognize Live Result:", res.json())
        assert res.status_code == 200
        
        res_ev = client.get("/api/events/live")
        events = res_ev.json()["events"]
        assert len(events) == 1
        assert events[0]["match_status"] == "MATCH"
        assert events[0]["matched_identity_id"] == "person_123"


def test_system_cleanup_endpoint(client):
    with patch("app.services.face_service.FaceService.run_storage_cleanup") as mock_cleanup:
        mock_cleanup.return_value = {
            "success": True,
            "deleted_evidence": 3,
            "deleted_recordings": 2,
            "retention_days": 30
        }
        res = client.post("/api/system/cleanup", json={"retention_days": 30})
        assert res.status_code == 200
        data = res.json()
        assert data["success"] is True
        assert data["deleted_evidence"] == 3
        assert data["deleted_recordings"] == 2

def test_health_storage_breakdown(client):
    with patch("app.services.face_service.FaceService.get_health_stats") as mock_health:
        mock_health.return_value = {
            "status": "healthy",
            "storage_used_bytes": 10240,
            "database_bytes": 2048,
            "evidence_bytes": 4096,
            "recordings_bytes": 3072,
            "reference_images_bytes": 1024
        }
        res = client.get("/api/health")
        assert res.status_code == 200
        data = res.json()
        assert data["storage_used_bytes"] == 10240
        assert data["database_bytes"] == 2048
        assert data["evidence_bytes"] == 4096
        assert data["recordings_bytes"] == 3072
        assert data["reference_images_bytes"] == 1024
