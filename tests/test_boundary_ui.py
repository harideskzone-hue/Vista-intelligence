"""
Boundary UI & End-to-End API Integration Tests (Block 3).

Verifies:
1. Serving boundary HTML page (/boundary) with rich Canvas UI, telemetry, and event feed.
2. Unified Navigation Dock across all pages (/dashboard, /preview, /wanted, /database, /boundary).
3. Boundary line and polygon zone configuration round-trip API with coordinate validation.
4. Rule-based anomaly threshold configuration and classification labels.
5. Zero filesystem path leakage in boundary event querying.
6. Surveillance video testing API (/api/boundary/test-video) with metric separation and clip streaming.
"""

import os
import sys
import json
import pytest
import cv2
import numpy as np
from unittest.mock import patch, MagicMock

# Preserve ultralytics if loaded, mock only insightface and bcrypt for light test startup
orig_ultralytics = sys.modules.get("ultralytics")
sys.modules["insightface"] = MagicMock()
sys.modules["insightface.app"] = MagicMock()
sys.modules["bcrypt"] = MagicMock()
if orig_ultralytics is not None:
    sys.modules["ultralytics"] = orig_ultralytics

from fastapi.testclient import TestClient
from edge.boundary.geometry import Point, Line, NormalizedLine
from edge.boundary.crossing_engine import TrackObservation
from edge.boundary.manager import get_boundary_manager
from edge.events.event_hub import get_event_hub


@pytest.fixture
def test_app(tmp_path):
    os.environ["SIH26187_DATA"] = str(tmp_path)
    
    # Mock cameras json
    cam_file = tmp_path / "cameras.json"
    cam_file.write_text("[]")
    import app.api.routes as routes
    routes._CAMERAS_JSON = str(cam_file)
    routes._live_events.clear()

    # Clear globals in face_service
    import app.services.face_service as fs
    fs._identity_manager = None
    fs._watchlist = None
    fs._temporal_matcher = None
    fs._face_matcher = None
    fs._best_frame_selector = None
    fs._duplicate_filter = None

    from face_api.run import create_app
    app = create_app()

    with patch("face_api.run.is_session_valid", return_value=True), \
         patch("app.services.face_service.get_storage"), \
         patch("app.services.face_service.get_vector_db"), \
         patch("app.services.face_service.get_encoder"), \
         patch("app.api.routes._validate_source", return_value=(True, "")):
        yield app


@pytest.fixture
def client(test_app):
    return TestClient(test_app)


# ── 1. Serve Boundary HTML Page ────────────────────────────────────────────────

def test_serve_boundary_page(client):
    res = client.get("/boundary")
    assert res.status_code == 200
    html = res.text
    # Verify core UI components exist
    assert "VISTA AI" in html
    assert "Border Vigil" in html or "Boundary" in html
    assert "boundary-canvas" in html
    assert "stat-crossed" in html
    assert "stat-intruding" in html
    assert "stat-retreating" in html
    assert "stat-zones" in html
    assert "stat-anomalies" in html
    assert "video-file-input" in html
    assert "video-modal" in html
    assert "Rule-Based Anomalies" in html or "Rule-Based Anomaly Indicators" in html


# ── 2. Unified Navigation Dock Across All Templates ────────────────────────────

def test_navigation_dock_across_pages(client):
    pages = ["/", "/preview", "/wanted", "/database", "/boundary"]
    for page in pages:
        res = client.get(page)
        assert res.status_code == 200, f"Failed to load {page}"
        html = res.text
        assert 'href="/boundary"' in html, f"/boundary link missing in {page}"
        assert "Border Vigil" in html, f"Border Vigil text missing in {page}"


# ── 3. Boundary Line & Polygon Zone Configuration API Roundtrip ────────────────

def test_boundary_line_and_zone_api_roundtrip(client):
    cam_id = "test_cam_dock_01"

    # Set normalized boundary line (horizontal crossing at y=0.5)
    res_line = client.post("/api/boundary/config", json={
        "camera_id": cam_id,
        "line": {"x1": 0.1, "y1": 0.5, "x2": 0.9, "y2": 0.5},
        "restricted_side": "A",
        "line_id": "primary_fence"
    })
    assert res_line.status_code == 200
    data_line = res_line.json()
    assert data_line["status"] == "success"
    assert data_line["config"]["line"]["x1"] == 0.1
    assert data_line["config"]["restricted_side"] == "A"

    # Get boundary line configuration
    res_get = client.get(f"/api/boundary/config?camera_id={cam_id}")
    assert res_get.status_code == 200
    cfg = res_get.json()
    assert cfg["camera_id"] == cam_id
    assert cfg["line"]["x1"] == 0.1
    assert cfg["line"]["y2"] == 0.5
    assert cfg["restricted_side"] == "A"

    # Test parameter validation: coordinates > 1.0 must fail with 422
    res_invalid = client.post("/api/boundary/config", json={
        "camera_id": cam_id,
        "line": {"x1": -0.5, "y1": 0.5, "x2": 1.5, "y2": 0.5},
        "restricted_side": "A"
    })
    assert res_invalid.status_code == 422

    # Set polygon zone configuration
    res_zone = client.post("/api/boundary/zones", json={
        "camera_id": cam_id,
        "zone_id": "buffer_corridor",
        "polygon": [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]]
    })
    assert res_zone.status_code == 200
    assert res_zone.json()["status"] == "success"

    # Delete polygon zone configuration
    res_del_zone = client.delete(f"/api/boundary/zones/{cam_id}/buffer_corridor")
    assert res_del_zone.status_code == 200


# ── 4. Rule-Based Anomaly Threshold Configuration API ──────────────────────────

def test_anomaly_thresholds_api(client):
    # Retrieve default thresholds
    res = client.get("/api/boundary/anomalies/config")
    assert res.status_code == 200
    data = res.json()
    assert data["classification"] == "Rule-based anomaly indicators"
    assert "running_speed_thresh" in data["thresholds"]
    assert "loitering_time" in data["thresholds"]

    # Reconfigure thresholds for specific camera
    cam_id = "test_cam_anomaly_ui"
    res_update = client.post("/api/boundary/anomalies/config", json={
        "camera_id": cam_id,
        "loitering_time": 12.0,
        "running_speed_thresh": 320.0
    })
    assert res_update.status_code == 200
    updated_data = res_update.json()
    assert updated_data["thresholds"]["loitering_time"] == 12.0
    assert updated_data["thresholds"]["running_speed_thresh"] == 320.0
    assert updated_data["classification"] == "Rule-based anomaly indicators"

    # Verify query for this camera reflects the updated thresholds
    res_check = client.get(f"/api/boundary/anomalies/config?camera_id={cam_id}")
    assert res_check.status_code == 200
    assert res_check.json()["thresholds"]["running_speed_thresh"] == 320.0


# ── 5. Zero Filesystem Path Leakage in Event Feeds ─────────────────────────────

def test_boundary_events_api_zero_path_leak(client):
    res = client.get("/api/boundary/events")
    assert res.status_code == 200
    events = res.json()
    assert isinstance(events, list)

    # If any events exist, verify internal server filesystem paths are never leaked
    for ev in events:
        assert "clip_path" not in ev
        assert "full_path" not in ev
        # Values should not contain absolute paths
        for k, v in ev.items():
            if isinstance(v, str):
                assert not v.startswith("/Users/"), f"Leaked local path in field {k}: {v}"
                assert not v.startswith("/home/"), f"Leaked local path in field {k}: {v}"


# ── 6. Surveillance Video Testing API & Operational Metrics ────────────────────

def test_surveillance_video_testing_api(client, tmp_path):
    # Generate a lightweight 6-frame MP4 video
    video_path = tmp_path / "test_surveillance_input.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_path), fourcc, 10.0, (400, 400))
    for i in range(6):
        out.write(np.full((400, 400, 3), 100, dtype=np.uint8))
    out.release()
    assert os.path.exists(str(video_path))

    # Mock tracking adapter to produce deterministic crossing and running behavior
    mock_adapter = MagicMock()
    mock_adapter.detect_and_track.side_effect = [
        [TrackObservation(track_id=1, bbox=(180, 50, 220, 100))],
        [TrackObservation(track_id=1, bbox=(180, 90, 220, 140))],
        [TrackObservation(track_id=1, bbox=(180, 220, 220, 270))],  # crosses line y=200
        [TrackObservation(track_id=1, bbox=(180, 280, 220, 330))],
        [TrackObservation(track_id=1, bbox=(180, 340, 220, 390))],
        [TrackObservation(track_id=1, bbox=(180, 350, 220, 400))],
    ]

    with patch("edge.boundary.adapter.get_tracking_adapter", return_value=mock_adapter):
        with open(str(video_path), "rb") as vf:
            res = client.post(
                "/api/boundary/test-video",
                files={"file": ("test_surveillance_input.mp4", vf, "video/mp4")},
                data={
                    "camera_id": "test_video_surveillance_ui",
                    "x1": 0.0,
                    "y1": 0.5,
                    "x2": 1.0,
                    "y2": 0.5,
                    "restricted_side": "A"
                }
            )

    assert res.status_code == 200, res.text
    text_lines = [l for l in res.text.strip().split("\n") if l.strip()]
    last_json = json.loads(text_lines[-1])
    data = last_json.get("results", last_json)
    assert data["status"] == "success"

    # Verify semantic separation of metrics
    assert "line_crossings" in data
    assert "zone_events" in data
    assert "anomalies" in data
    assert "persons_crossed" in data

    assert data["line_crossings"]["total"] == 1
    assert data["line_crossings"]["intruding"] == 1
    assert data["line_crossings"]["retreating"] == 0
    assert data["zone_events"]["entered"] == 0
    assert data["zone_events"]["exited"] == 0
    assert data["persons_crossed"] == data["line_crossings"]["total"]

    # Verify event listings and clip generation
    assert "crossing_events" in data
    assert len(data["crossing_events"]) == 1
    assert data["crossing_events"][0]["direction"] == "INTRUDING"

    # Query event hub to verify generated clip is streamable
    hub = get_event_hub()
    events = hub.list_events(camera_id="test_video_surveillance_ui")
    assert len(events) >= 1

    ev = events[0]
    assert ev.clip_status == "READY"
    assert ev.clip_path is not None
    assert os.path.exists(ev.clip_path)

    # Test clip streaming route
    res_clip = client.get(f"/api/events/{ev.id}/clip")
    assert res_clip.status_code in (200, 206)
    assert res_clip.headers.get("content-type") == "video/mp4"
