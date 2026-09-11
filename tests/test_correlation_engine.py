import time
import uuid
import pytest

from edge.events.event_hub import get_event_hub, SystemEvent
from edge.events.correlation import get_correlation_engine

@pytest.fixture(autouse=True)
def setup_teardown():
    hub = get_event_hub()
    with hub._lock:
        with hub._get_conn() as conn:
            conn.execute("DELETE FROM system_events")
    hub._live_events.clear()
    
    engine = get_correlation_engine()
    engine.camera_buffers.clear()
    engine.dedup_cache.clear()
    
    yield

def test_1_face_vehicle_within_window_correlation():
    hub = get_event_hub()
    engine = get_correlation_engine()
    
    # Send FACE event
    ev1 = hub.log_event("FACE_DETECTED", "FACE", "CAM1", timestamp=100.0, details="Face", metadata={"zone_id": "zoneA"})
    
    # Send VEHICLE event within window (5 seconds)
    ev2 = hub.log_event("VEHICLE_DETECTED", "VEHICLE", "CAM1", timestamp=102.0, details="Vehicle", metadata={"zone_id": "zoneA"})
    
    # Give async a tiny bit of time to run
    time.sleep(0.1)
    
    events = list(hub._live_events)
    correlations = [e for e in events if e.subsystem == "CORRELATION"]
    assert len(correlations) == 1
    assert correlations[0].metadata["correlation_type"] == "FACE_VEHICLE_PROXIMITY"
    assert ev1.id in correlations[0].metadata["source_event_ids"]
    assert ev2.id in correlations[0].metadata["source_event_ids"]

def test_2_same_events_outside_window():
    hub = get_event_hub()
    
    hub.log_event("FACE_DETECTED", "FACE", "CAM1", timestamp=100.0, details="Face")
    hub.log_event("VEHICLE_DETECTED", "VEHICLE", "CAM1", timestamp=110.0, details="Vehicle") # 10s gap, > 5s
    
    time.sleep(0.1)
    events = list(hub._live_events)
    correlations = [e for e in events if e.subsystem == "CORRELATION"]
    assert len(correlations) == 0

def test_3_different_cameras():
    hub = get_event_hub()
    
    hub.log_event("FACE_DETECTED", "FACE", "CAM1", timestamp=100.0)
    hub.log_event("VEHICLE_DETECTED", "VEHICLE", "CAM2", timestamp=102.0)
    
    time.sleep(0.1)
    events = list(hub._live_events)
    correlations = [e for e in events if e.subsystem == "CORRELATION"]
    assert len(correlations) == 0

def test_4_source_events_unmodified():
    hub = get_event_hub()
    ev1 = hub.log_event("FACE_DETECTED", "FACE", "CAM1", timestamp=100.0)
    ev2 = hub.log_event("VEHICLE_DETECTED", "VEHICLE", "CAM1", timestamp=102.0)
    
    time.sleep(0.1)
    # Check original events didn't get modified by engine
    # In SQLite
    with hub._get_conn() as conn:
        res = conn.execute("SELECT subsystem, details FROM system_events WHERE id=?", (ev1.id,)).fetchone()
        assert res["subsystem"] == "FACE"

def test_5_correlation_creates_new_event():
    hub = get_event_hub()
    hub.log_event("FACE_DETECTED", "FACE", "CAM1", timestamp=100.0)
    hub.log_event("VEHICLE_DETECTED", "VEHICLE", "CAM1", timestamp=102.0)
    
    time.sleep(0.1)
    with hub._get_conn() as conn:
        c = conn.execute("SELECT COUNT(*) as c FROM system_events WHERE subsystem='CORRELATION'").fetchone()
        assert c["c"] == 1

def test_6_duplicate_source_pair():
    hub = get_event_hub()
    ev1 = hub.log_event("FACE_DETECTED", "FACE", "CAM1", timestamp=100.0)
    ev2 = hub.log_event("VEHICLE_DETECTED", "VEHICLE", "CAM1", timestamp=102.0)
    
    time.sleep(0.1)
    # Re-evaluate
    engine = get_correlation_engine()
    engine._evaluate_face_vehicle(ev2, engine.camera_buffers["CAM1"])
    
    time.sleep(0.1)
    events = list(hub._live_events)
    correlations = [e for e in events if e.subsystem == "CORRELATION"]
    assert len(correlations) == 1  # Deduplicated

def test_7_unknown_identity_safe():
    hub = get_event_hub()
    hub.log_event("FACE_DETECTED", "FACE", "CAM1", timestamp=100.0, person_id="UNKNOWN")
    hub.log_event("VEHICLE_DETECTED", "VEHICLE", "CAM1", timestamp=102.0, metadata={"plate_text": "UNKNOWN"})
    
    time.sleep(0.1)
    events = list(hub._live_events)
    correlations = [e for e in events if e.subsystem == "CORRELATION"]
    assert len(correlations) == 1
    assert "Unknown person and unknown vehicle" in correlations[0].details

def test_8_anpr_match_boundary_alert():
    hub = get_event_hub()
    hub.log_event("ANPR_MATCH", "VEHICLE", "CAM1", timestamp=100.0, metadata={"plate_text": "STOLEN1"})
    hub.log_event("BOUNDARY_CROSSED", "BOUNDARY", "CAM1", timestamp=102.0)
    
    time.sleep(0.1)
    events = list(hub._live_events)
    correlations = [e for e in events if e.subsystem == "CORRELATION" and e.event_type == "ALERT"]
    assert len(correlations) == 1
    assert correlations[0].severity == "CRITICAL"

def test_9_face_match_boundary_alert():
    hub = get_event_hub()
    hub.log_event("FACE_MATCH", "FACE", "CAM1", timestamp=100.0, person_id="WANTED_GUY")
    hub.log_event("ANOMALY", "BOUNDARY", "CAM1", timestamp=102.0)
    
    time.sleep(0.1)
    events = list(hub._live_events)
    correlations = [e for e in events if e.subsystem == "CORRELATION" and e.event_type == "ALERT"]
    assert len(correlations) == 1

def test_10_spatial_metadata_improves_correlation():
    hub = get_event_hub()
    hub.log_event("FACE_DETECTED", "FACE", "CAM1", timestamp=100.0, metadata={"bounding_box": [1,2,3,4]})
    hub.log_event("VEHICLE_DETECTED", "VEHICLE", "CAM1", timestamp=102.0, metadata={"bounding_box": [1,2,3,4]})
    
    time.sleep(0.1)
    events = list(hub._live_events)
    correlations = [e for e in events if e.subsystem == "CORRELATION"]
    assert correlations[0].metadata["confidence_level"] == "STRONG"

def test_11_missing_spatial_graceful_fallback():
    hub = get_event_hub()
    hub.log_event("FACE_DETECTED", "FACE", "CAM1", timestamp=100.0, metadata={})
    hub.log_event("VEHICLE_DETECTED", "VEHICLE", "CAM1", timestamp=102.0, metadata={})
    
    time.sleep(0.1)
    events = list(hub._live_events)
    correlations = [e for e in events if e.subsystem == "CORRELATION"]
    assert correlations[0].metadata["confidence_level"] == "WEAK"

def test_12_multicamera_isolation():
    hub = get_event_hub()
    hub.log_event("FACE_DETECTED", "FACE", "CAM1", timestamp=100.0)
    hub.log_event("VEHICLE_DETECTED", "VEHICLE", "CAM2", timestamp=102.0)
    hub.log_event("FACE_DETECTED", "FACE", "CAM2", timestamp=100.0)
    hub.log_event("VEHICLE_DETECTED", "VEHICLE", "CAM1", timestamp=102.0)
    
    time.sleep(0.1)
    events = list(hub._live_events)
    correlations = [e for e in events if e.subsystem == "CORRELATION"]
    # Should get 2 correlations: CAM1<->CAM1 and CAM2<->CAM2
    assert len(correlations) == 2
    cameras = {e.camera_id for e in correlations}
    assert "CAM1" in cameras and "CAM2" in cameras

def test_13_nonblocking_dispatch():
    # If the engine sleeps, log_event shouldn't wait
    hub = get_event_hub()
    engine = get_correlation_engine()
    
    def slow_subscriber(event):
        time.sleep(0.2)
        
    hub.subscribe(slow_subscriber)
    
    start = time.time()
    hub.log_event("FACE_DETECTED", "FACE", "CAM1")
    assert time.time() - start < 0.1 # Should be fast
    hub._subscribers.remove(slow_subscriber)

def test_14_correlation_survives_restart():
    hub = get_event_hub()
    hub.log_event("FACE_DETECTED", "FACE", "CAM1", timestamp=100.0)
    hub.log_event("VEHICLE_DETECTED", "VEHICLE", "CAM1", timestamp=102.0)
    
    time.sleep(0.1)
    
    # simulate restart by checking sqlite
    with hub._get_conn() as conn:
        res = conn.execute("SELECT id FROM system_events WHERE subsystem='CORRELATION'").fetchall()
        assert len(res) == 1

def test_15_api_shows_correlation_events():
    from fastapi.testclient import TestClient
    from face_api.run import app
    client = TestClient(app)
    
    hub = get_event_hub()
    hub.log_event("FACE_DETECTED", "FACE", "CAM1", timestamp=100.0)
    hub.log_event("VEHICLE_DETECTED", "VEHICLE", "CAM1", timestamp=102.0)
    time.sleep(0.1)
    
    # Query API
    res = client.get("/api/events?subsystem=CORRELATION")
    assert res.status_code == 200
    events = res.json()
    assert len(events) == 1

def test_16_secure_clip_path():
    hub = get_event_hub()
    hub.log_event("FACE_DETECTED", "FACE", "CAM1", timestamp=100.0)
    hub.log_event("VEHICLE_DETECTED", "VEHICLE", "CAM1", timestamp=102.0)
    time.sleep(0.1)
    
    from fastapi.testclient import TestClient
    from face_api.run import app
    client = TestClient(app)
    res = client.get("/api/events?subsystem=CORRELATION")
    events = res.json()
    assert len(events) == 1
    assert str(events[0].get("clip_path", "")).startswith("/") is False # Assuming None or empty since generate_clip=False
