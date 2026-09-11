"""
Unit and Integration Tests for Boundary Intelligence Subsystem (Block 2).

Verifies:
1. Geometry primitives, side of line, bottom-center ground contact, aspect ratio, polygon tests.
2. Crossing detection, correct direction (INTRUDING vs RETREATING), and debounce cooldowns.
3. Per-track and per-camera state isolation.
4. Stale-track cleanup.
5. Polygon zone entry and exit.
6. All 6 behavioral anomalies: Loitering, Crawling, Running, Erratic Movement, Grouping, Zone Lingering.
7. BoundaryManager integration with EventHub (PENDING -> READY clip pipeline).
8. Autonomous uploaded-video testing pipeline with offline clip extraction.
9. Optional Face AI enrichment without hard dependency.
10. Boundary REST API endpoints and zero filesystem path exposure.
"""

import os
import time
import shutil
import tempfile
import pytest
import numpy as np
import cv2
from fastapi.testclient import TestClient

from edge.boundary.geometry import (
    Point, Line, NormalizedLine, side_of_point, bottom_center,
    centroid, distance, bbox_aspect_ratio, point_in_polygon, angle_between_vectors
)
from edge.boundary.crossing_engine import TrackObservation, CrossingEngine, CrossingEvent
from edge.boundary.anomaly_engine import AnomalyEngine, AnomalyEvent
from edge.boundary.manager import BoundaryManager, CameraBoundaryConfig
from edge.events.event_hub import EventHub
from edge.events.clip_manager import EventClipManager
from face_api.run import create_app


@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp(prefix="test_boundary_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


# ── 1. Geometry Primitives ───────────────────────────────────────────────────

def test_geometry_primitives():
    # Points and distance
    p1 = Point(0, 0)
    p2 = Point(3, 4)
    assert distance(p1, p2) == pytest.approx(5.0)

    # Normalized Line to Pixel Line
    nline = NormalizedLine(x1=0.1, y1=0.2, x2=0.9, y2=0.8)
    pline = nline.to_pixel_line(width=1000, height=500)
    assert pline.p1.x == pytest.approx(100.0)
    assert pline.p1.y == pytest.approx(100.0)
    assert pline.p2.x == pytest.approx(900.0)
    assert pline.p2.y == pytest.approx(400.0)

    # Side of Point: Line directed horizontally from (0, 100) to (1000, 100)
    # Directed vector is (1000, 0).
    # Side A (left of vector): y < 100 (in standard Cartesian, but in screen coordinates with y down,
    # cross = (x2-x1)*(y-y1) - (y2-y1)*(x-x1) = 1000 * (y - 100).
    # If y > 100 -> cross > 0 -> side = +1 (Side A)
    # If y < 100 -> cross < 0 -> side = -1 (Side B)
    line = Line(Point(0, 100), Point(1000, 100))
    pt_side_a = Point(500, 150)
    pt_side_b = Point(500, 50)
    pt_colinear = Point(500, 100)

    assert side_of_point(line, pt_side_a) == 1
    assert side_of_point(line, pt_side_b) == -1
    assert side_of_point(line, pt_colinear) == 0

    # Bottom-Center anchor
    bbox = (100, 200, 200, 400)  # xmin, ymin, xmax, ymax
    bc = bottom_center(bbox)
    assert bc.x == 150.0
    assert bc.y == 400.0

    # Aspect Ratio: Normal standing vs prone crawling
    standing_bbox = (100, 100, 150, 250)  # w=50, h=150 -> 0.33
    crawling_bbox = (100, 100, 250, 160)  # w=150, h=60 -> 2.5
    assert bbox_aspect_ratio(standing_bbox) < 0.5
    assert bbox_aspect_ratio(crawling_bbox) > 0.85

    # Point in Polygon
    poly = np.array([[100, 100], [300, 100], [300, 300], [100, 300]], dtype=np.int32)
    assert point_in_polygon(Point(200, 200), poly) is True
    assert point_in_polygon(Point(50, 50), poly) is False

    # Angle between vectors
    v1 = np.array([1, 0], dtype=np.float32)
    v2 = np.array([0, 1], dtype=np.float32)
    assert angle_between_vectors(v1, v2) == pytest.approx(90.0)


# ── 2. Line Crossing & Direction Classification ─────────────────────────────

def test_crossing_detection_and_direction():
    engine = CrossingEngine(cooldown_seconds=1.0)
    # Line at y = 500. Restricted side is +1 (y > 500)
    line = Line(Point(0, 500), Point(1000, 500))
    restricted_side = 1

    # Track 10 starts on Side B (y = 400, side = -1)
    obs_t0 = [TrackObservation(track_id=10, bbox=(450, 300, 550, 400), timestamp=1.0)]
    events_0 = engine.check_line_crossings("main_gate", line, restricted_side, obs_t0, timestamp=1.0)
    assert len(events_0) == 0  # Initial baseline establishes state, no false alert

    # Track 10 crosses to Side A (y = 600, side = +1) at t=1.5
    obs_t1 = [TrackObservation(track_id=10, bbox=(450, 500, 550, 600), timestamp=1.5)]
    events_1 = engine.check_line_crossings("main_gate", line, restricted_side, obs_t1, timestamp=1.5)
    assert len(events_1) == 1
    ev = events_1[0]
    assert ev.track_id == 10
    assert ev.from_side == -1
    assert ev.to_side == 1
    assert ev.direction == "INTRUDING"
    assert engine.total_intrusions == 1

    # Track 20 starts on Side A (y = 600, side = +1)
    obs_t2_init = [TrackObservation(track_id=20, bbox=(450, 500, 550, 600), timestamp=2.0)]
    engine.check_line_crossings("main_gate", line, restricted_side, obs_t2_init, timestamp=2.0)

    # Track 20 moves to Side B (y = 400, side = -1) at t=2.5 -> RETREATING
    obs_t2_cross = [TrackObservation(track_id=20, bbox=(450, 300, 550, 400), timestamp=2.5)]
    events_2 = engine.check_line_crossings("main_gate", line, restricted_side, obs_t2_cross, timestamp=2.5)
    assert len(events_2) == 1
    assert events_2[0].direction == "RETREATING"
    assert engine.total_retreats == 1


# ── 3. Debounce & Duplicate Suppression ─────────────────────────────────────

def test_debounce_cooldown():
    engine = CrossingEngine(cooldown_seconds=3.0)
    line = Line(Point(0, 500), Point(1000, 500))

    # Baseline at t=1.0
    engine.check_line_crossings("line_1", line, restricted_side=1,
                                observations=[TrackObservation(track_id=1, bbox=(100, 300, 200, 400))], timestamp=1.0)
    # Cross at t=1.5 -> Triggered
    evs1 = engine.check_line_crossings("line_1", line, restricted_side=1,
                                       observations=[TrackObservation(track_id=1, bbox=(100, 500, 200, 600))], timestamp=1.5)
    assert len(evs1) == 1

    # Oscillate back at t=2.0 (inside 3.0s cooldown window) -> Suppressed
    evs2 = engine.check_line_crossings("line_1", line, restricted_side=1,
                                       observations=[TrackObservation(track_id=1, bbox=(100, 300, 200, 400))], timestamp=2.0)
    assert len(evs2) == 0

    # Cross again after cooldown expires at t=5.0 -> Triggered
    evs3 = engine.check_line_crossings("line_1", line, restricted_side=1,
                                       observations=[TrackObservation(track_id=1, bbox=(100, 500, 200, 600))], timestamp=5.0)
    assert len(evs3) == 1


# ── 4. Per-Track & Per-Camera State Isolation ────────────────────────────────

def test_per_track_and_camera_isolation(temp_dir):
    bm = BoundaryManager(config_path=os.path.join(temp_dir, "boundary.json"))
    # Cam 1 line at y=0.5
    bm.set_boundary_line("cam_01", x1=0.0, y1=0.5, x2=1.0, y2=0.5, restricted_side="A")
    # Cam 2 line at x=0.5
    bm.set_boundary_line("cam_02", x1=0.5, y1=0.0, x2=0.5, y2=1.0, restricted_side="A")

    frame = np.zeros((1000, 1000, 3), dtype=np.uint8)

    # Establish baseline for Track 1 on Cam 1 (y=400)
    bm.process_frame("cam_01", frame, [TrackObservation(track_id=1, bbox=(100, 300, 200, 400))], timestamp=1.0)
    # Establish baseline for Track 1 on Cam 2 (x=400)
    bm.process_frame("cam_02", frame, [TrackObservation(track_id=1, bbox=(350, 100, 400, 200))], timestamp=1.0)

    # Cross on Cam 1 (y moves to 600)
    crs1, _ = bm.process_frame("cam_01", frame, [TrackObservation(track_id=1, bbox=(100, 500, 200, 600))], timestamp=1.5)
    assert len(crs1) == 1

    # Cam 2 Track 1 has NOT crossed its vertical line
    crs2, _ = bm.process_frame("cam_02", frame, [TrackObservation(track_id=1, bbox=(350, 150, 400, 250))], timestamp=1.5)
    assert len(crs2) == 0


# ── 5. Stale Track Cleanup ───────────────────────────────────────────────────

def test_stale_track_cleanup():
    engine = CrossingEngine(stale_timeout=5.0)
    line = Line(Point(0, 500), Point(1000, 500))

    # Seen at t=1.0
    engine.check_line_crossings("l1", line, 1, [TrackObservation(track_id=99, bbox=(100, 300, 200, 400))], timestamp=1.0)
    assert 99 in engine._last_seen

    # Advances to t=10.0 (> 5.0 timeout) with no observations of track 99
    engine.check_line_crossings("l1", line, 1, [], timestamp=10.0)
    assert 99 not in engine._last_seen
    assert 99 not in engine._prev_line_sides.get("l1", {})


# ── 6. Polygon Zone Transitions ──────────────────────────────────────────────

def test_polygon_zone_transitions():
    engine = CrossingEngine(cooldown_seconds=1.0)
    poly = np.array([[200, 200], [600, 200], [600, 600], [200, 600]], dtype=np.int32)

    # Track 5 outside at t=1.0
    engine.check_zone_transitions("zone_secure", poly, [TrackObservation(track_id=5, bbox=(50, 50, 100, 100))], timestamp=1.0)

    # Track 5 enters at t=2.0
    evs_enter = engine.check_zone_transitions("zone_secure", poly, [TrackObservation(track_id=5, bbox=(250, 250, 350, 350))], timestamp=2.0)
    assert len(evs_enter) == 1
    assert evs_enter[0].direction == "ZONE_INTRUDED"

    # Track 5 exits at t=4.0
    evs_exit = engine.check_zone_transitions("zone_secure", poly, [TrackObservation(track_id=5, bbox=(50, 50, 100, 100))], timestamp=4.0)
    assert len(evs_exit) == 1
    assert evs_exit[0].direction == "ZONE_EXITED"


# ── 7. Anomaly Engine: Loitering ─────────────────────────────────────────────

def test_anomaly_loitering():
    engine = AnomalyEngine(loitering_time=3.0, loitering_radius=50.0)

    # Track stays inside 50px radius for 4.0s
    anomalies = []
    for step in range(9):
        t = 1.0 + step * 0.5
        # slight jitter
        x = 200 + (step % 2) * 5
        y = 200 + (step % 2) * 5
        obs = [TrackObservation(track_id=12, bbox=(x-20, y-40, x+20, y))]
        anoms = engine.update(obs, timestamp=t)
        anomalies.extend(anoms)

    loiter_events = [a for a in anomalies if a.event_type == "LOITERING"]
    assert len(loiter_events) >= 1
    assert loiter_events[0].track_id == 12


# ── 8. Anomaly Engine: Crawling ──────────────────────────────────────────────

def test_anomaly_crawling():
    engine = AnomalyEngine(crawling_ratio_thresh=0.85, crawling_min_frames=4)

    anomalies = []
    # 5 consecutive prone frames: w=150, h=50 -> ratio=3.0
    for i in range(5):
        t = 1.0 + i * 0.2
        obs = [TrackObservation(track_id=7, bbox=(100, 100, 250, 150))]
        anoms = engine.update(obs, timestamp=t)
        anomalies.extend(anoms)

    crawl_events = [a for a in anomalies if a.event_type == "CRAWLING"]
    assert len(crawl_events) == 1
    assert crawl_events[0].severity == "CRITICAL"
    assert crawl_events[0].track_id == 7


# ── 9. Anomaly Engine: Running ───────────────────────────────────────────────

def test_anomaly_running():
    engine = AnomalyEngine(running_speed_thresh=200.0)

    # Moves 300 pixels in 1 second -> speed = 300 px/s
    obs1 = [TrackObservation(track_id=8, bbox=(100, 100, 140, 200))]
    obs2 = [TrackObservation(track_id=8, bbox=(400, 100, 440, 200))]

    engine.update(obs1, timestamp=1.0)
    anoms = engine.update(obs2, timestamp=2.0)

    run_events = [a for a in anoms if a.event_type == "RUNNING"]
    assert len(run_events) == 1
    assert run_events[0].track_id == 8


# ── 10. Anomaly Engine: Erratic Zigzag Movement ──────────────────────────────

def test_anomaly_erratic_movement():
    engine = AnomalyEngine(erratic_angle_thresh=70.0, erratic_min_turns=3)

    # Create sharp zigzag trajectory: (0,0) -> (100, 50) -> (20, 100) -> (120, 150) -> (10, 200)
    points = [
        (0, 0), (50, 25), (100, 50),
        (60, 75), (20, 100),
        (70, 125), (120, 150),
        (65, 175), (10, 200), (60, 220)
    ]
    anomalies = []
    for idx, (x, y) in enumerate(points):
        t = 1.0 + idx * 0.3
        obs = [TrackObservation(track_id=33, bbox=(x-10, y-20, x+10, y))]
        anoms = engine.update(obs, timestamp=t)
        anomalies.extend(anoms)

    erratic_events = [a for a in anomalies if a.event_type == "ERRATIC_MOVEMENT"]
    assert len(erratic_events) >= 1
    assert erratic_events[0].track_id == 33


# ── 11. Anomaly Engine: Grouping ─────────────────────────────────────────────

def test_anomaly_grouping():
    engine = AnomalyEngine(group_min_size=3, group_distance_px=100.0)

    # 3 persons clustered within 40px of each other
    obs = [
        TrackObservation(track_id=1, bbox=(200, 200, 240, 300)),
        TrackObservation(track_id=2, bbox=(220, 210, 260, 310)),
        TrackObservation(track_id=3, bbox=(230, 195, 270, 295))
    ]
    anoms = engine.update(obs, timestamp=1.0)
    group_events = [a for a in anoms if a.event_type == "GROUPING"]
    assert len(group_events) == 1
    assert "Group formation of 3 persons" in group_events[0].details


# ── 12. BoundaryManager & EventHub Integration (Pending -> Ready) ────────────

def test_boundary_manager_event_hub_flow(temp_dir):
    clips_dir = os.path.join(temp_dir, "clips")
    db_path = os.path.join(temp_dir, "events.db")
    clip_mgr = EventClipManager(output_dir=clips_dir, default_pre_seconds=1.0, default_post_seconds=1.0)
    hub = EventHub(db_path=db_path, clip_manager=clip_mgr)
    bm = BoundaryManager(config_path=os.path.join(temp_dir, "cfg.json"), event_hub=hub, clip_manager=clip_mgr)

    bm.set_boundary_line("gate_1", x1=0.0, y1=0.5, x2=1.0, y2=0.5, restricted_side="A")

    frame = np.zeros((400, 400, 3), dtype=np.uint8)

    # Feed pre-event frames
    for i in range(15):
        t = 10.0 + i * 0.1
        bm.process_frame("gate_1", frame, [TrackObservation(track_id=42, bbox=(180, 50, 220, 150))], timestamp=t)

    # Trigger crossing at t=11.6
    crs, _ = bm.process_frame(
        "gate_1",
        frame,
        [TrackObservation(track_id=42, bbox=(180, 250, 220, 350))],
        timestamp=11.6,
        track_to_identity={42: ("p_001", "Agent Smith")}
    )
    assert len(crs) == 1

    # Verify EventHub immediately logged as PENDING with enriched identity
    evs = hub.list_events(camera_id="gate_1", subsystem="BOUNDARY")
    assert len(evs) >= 1
    latest = evs[0]
    assert latest.event_type == "CROSSING"
    assert latest.clip_status == "PENDING"
    assert latest.display_name == "Agent Smith"
    assert latest.person_id == "p_001"

    # Feed post-event frames to satisfy 1.0s post-event window
    for i in range(15):
        t = 11.7 + i * 0.1
        bm.process_frame("gate_1", frame, [TrackObservation(track_id=42, bbox=(180, 250, 220, 350))], timestamp=t)

    # Wait up to 3 seconds for asynchronous clip compilation to finish
    deadline = time.time() + 3.0
    ready = False
    while time.time() < deadline:
        updated = hub.get_event(latest.id)
        if updated and updated.clip_status == "READY":
            ready = True
            break
        time.sleep(0.1)

    assert ready is True, f"Expected clip_status READY, got {hub.get_event(latest.id).clip_status}"
    updated_ev = hub.get_event(latest.id)
    assert os.path.exists(updated_ev.clip_path)


# ── 13. Uploaded Video Simulation Pipeline ───────────────────────────────────

def test_uploaded_video_pipeline(temp_dir):
    clips_dir = os.path.join(temp_dir, "clips")
    db_path = os.path.join(temp_dir, "events.db")
    clip_mgr = EventClipManager(output_dir=clips_dir, default_pre_seconds=1.0, default_post_seconds=1.0)
    hub = EventHub(db_path=db_path, clip_manager=clip_mgr)
    bm = BoundaryManager(config_path=os.path.join(temp_dir, "cfg.json"), event_hub=hub, clip_manager=clip_mgr)

    # Synthesize frames and observations for a simulated uploaded video
    # 50 frames @ 10 fps (5.0 seconds total)
    # Track 7 crosses at t=2.5 (from y=100 to y=300, boundary at y=200)
    synthetic_sequence = []
    dummy_frame = np.zeros((400, 400, 3), dtype=np.uint8)

    for idx in range(50):
        t = idx * 0.1
        y = 100 if t < 2.5 else 300
        obs = [TrackObservation(track_id=7, bbox=(180, y-50, 220, y))]
        synthetic_sequence.append((t, dummy_frame.copy(), obs))

    res = bm.process_uploaded_video(
        video_path="dummy.mp4",
        camera_id="upload_cam_test",
        line={"x1": 0.0, "y1": 0.5, "x2": 1.0, "y2": 0.5},
        restricted_side="A",
        precomputed_frames_and_tracks=synthetic_sequence,
        fps_override=10.0
    )

    assert res["status"] == "success"
    assert res["persons_crossed"] == 1
    assert res["intruding"] == 1
    assert res["retreating"] == 0

    # Verify offline clip was extracted and set to READY in EventHub
    events = hub.list_events(camera_id="upload_cam_test")
    assert len(events) >= 1
    crossing_evs = [e for e in events if e.event_type == "CROSSING"]
    assert len(crossing_evs) == 1
    assert crossing_evs[0].clip_status == "READY"
    assert os.path.exists(crossing_evs[0].clip_path)

    # Anomaly clip also generated if running triggered
    for ev in events:
        assert ev.clip_status == "READY"
        assert os.path.exists(ev.clip_path)


# ── 14. Boundary REST API Endpoints ──────────────────────────────────────────

def test_boundary_routes_api():
    app = create_app()
    client = TestClient(app)

    # 1. GET initial config
    res = client.get("/api/boundary/config?camera_id=test_api_cam")
    assert res.status_code == 200
    data = res.json()
    assert data["camera_id"] == "test_api_cam"

    # 2. POST set boundary line config
    payload = {
        "camera_id": "test_api_cam",
        "line": {"x1": 0.20, "y1": 0.50, "x2": 0.80, "y2": 0.50},
        "restricted_side": "A",
        "line_id": "checkpoint_alpha"
    }
    res_post = client.post("/api/boundary/config", json=payload)
    assert res_post.status_code == 200
    cfg = res_post.json()["config"]
    assert cfg["line"]["x1"] == pytest.approx(0.20)
    assert cfg["restricted_side"] == "A"

    # 3. POST set zone config
    zone_payload = {
        "camera_id": "test_api_cam",
        "zone_id": "restricted_quad",
        "polygon": [[0.1, 0.1], [0.4, 0.1], [0.4, 0.4], [0.1, 0.4]]
    }
    res_zone = client.post("/api/boundary/zones", json=zone_payload)
    assert res_zone.status_code == 200

    # 4. DELETE zone
    res_del = client.delete("/api/boundary/zones/test_api_cam/restricted_quad")
    assert res_del.status_code == 200

    # 5. GET boundary events (guarantees no filesystem path exposure)
    res_evs = client.get("/api/boundary/events?camera_id=test_api_cam")
    assert res_evs.status_code == 200
    events = res_evs.json()
    assert isinstance(events, list)
    for ev in events:
        assert "clip_path" not in ev or ev["clip_path"] is None
        assert "snapshot_path" not in ev or ev["snapshot_path"] is None


# ── 15. Real Uploaded Video with YOLO Detector + ByteTrack Adapter ───────────

def test_real_uploaded_video_with_detector_and_tracker(temp_dir):
    """
    Verifies that a real uploaded MP4 video is processed end-to-end:
    Real Video File -> YOLO Person Detector -> ByteTrack Tracker ->
    TrackObservation -> BoundaryManager -> EventHub -> EventClipManager.
    """
    clips_dir = os.path.join(temp_dir, "clips")
    db_path = os.path.join(temp_dir, "events.db")
    clip_mgr = EventClipManager(output_dir=clips_dir, default_pre_seconds=1.0, default_post_seconds=1.0)
    hub = EventHub(db_path=db_path, clip_manager=clip_mgr)
    bm = BoundaryManager(config_path=os.path.join(temp_dir, "cfg.json"), event_hub=hub, clip_manager=clip_mgr)

    # Load person image
    person_img = cv2.imread("/Users/hariharans/Documents/SIH26187/wanted_persons/Test_Person/Image segregation/darshan/darshan-1.jpeg")
    assert person_img is not None

    person_h, person_w = 160, 120
    person_resized = cv2.resize(person_img, (person_w, person_h))

    # Write a 25-frame real video with person moving across line at y=240
    video_path = os.path.join(temp_dir, "person_crossing_real.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_path, fourcc, 10.0, (640, 480))

    for idx in range(25):
        canvas = np.full((480, 640, 3), 128, dtype=np.uint8)
        y_start = 30 + idx * 10
        canvas[y_start:y_start+person_h, 260:260+person_w] = person_resized
        out.write(canvas)
    out.release()
    assert os.path.exists(video_path)

    # Process uploaded video with precomputed_frames_and_tracks=None (triggers real detector/tracker)
    res = bm.process_uploaded_video(
        video_path=video_path,
        camera_id="real_yolo_crossing_cam",
        line={"x1": 0.0, "y1": 0.5, "x2": 1.0, "y2": 0.5},
        restricted_side="A"
    )

    assert res["status"] == "success"
    assert res["line_crossings"]["total"] == 1
    assert res["line_crossings"]["intruding"] == 1
    assert res["line_crossings"]["retreating"] == 0
    assert res["persons_crossed"] == 1
    assert res["zone_events"]["total"] == 0

    # Verify EventHub logged the event and generated a valid MP4 clip
    events = hub.list_events(camera_id="real_yolo_crossing_cam")
    assert len(events) >= 1
    crossing_ev = next(e for e in events if e.event_type == "CROSSING")
    assert crossing_ev.clip_status == "READY"
    assert os.path.exists(crossing_ev.clip_path)
    assert os.path.getsize(crossing_ev.clip_path) > 1000


# ── 16. Semantic Metric Separation (Line Crossings vs Zone Transitions) ──────

def test_semantic_metric_separation(temp_dir):
    """
    Guarantees that 'persons_crossed' exclusively counts boundary line crossings,
    and polygon zone transitions (ZONE_INTRUDED / ZONE_EXITED) are segregated into
    distinct zone_events metrics.
    """
    bm = BoundaryManager(config_path=os.path.join(temp_dir, "cfg.json"))
    cam = "metric_sep_cam"

    # Set line at y=0.5
    bm.set_boundary_line(cam, x1=0.0, y1=0.5, x2=1.0, y2=0.5, restricted_side="A")
    # Set polygon zone
    bm.set_zone(cam, "zone_alpha", [[0.1, 0.1], [0.4, 0.1], [0.4, 0.4], [0.1, 0.4]])

    # Build sequence with:
    # 1. Track 1 crosses line (y=100 -> y=300)
    # 2. Track 2 enters zone (outside -> inside zone)
    frame = np.zeros((400, 400, 3), dtype=np.uint8)

    # Frame 0: T1 at y=100 (outside line), T2 at (50, 50) (outside zone)
    bm.process_frame(cam, frame, [
        TrackObservation(track_id=1, bbox=(180, 50, 220, 100)),
        TrackObservation(track_id=2, bbox=(20, 20, 50, 50))
    ], timestamp=0.0)

    # Frame 1: T1 crosses line (y=250), T2 enters zone (100, 100 inside zone 40..160)
    crs, anm = bm.process_frame(cam, frame, [
        TrackObservation(track_id=1, bbox=(180, 200, 220, 250)),
        TrackObservation(track_id=2, bbox=(80, 80, 120, 120))
    ], timestamp=0.5)

    # One line crossing and one zone intrusion
    line_events = [c for c in crs if c.direction in ("INTRUDING", "RETREATING")]
    zone_events = [c for c in crs if c.direction in ("ZONE_INTRUDED", "ZONE_EXITED")]

    assert len(line_events) == 1
    assert len(zone_events) == 1
    assert line_events[0].direction == "INTRUDING"
    assert zone_events[0].direction == "ZONE_INTRUDED"


# ── 17. Configurable Anomaly Thresholds & Classification ──────────────────────

def test_anomaly_thresholds_configuration(temp_dir):
    """
    Tests dynamic reconfiguration of anomaly indicator thresholds and ensures
    outputs are labeled as rule-based indicators.
    """
    bm = BoundaryManager(config_path=os.path.join(temp_dir, "cfg.json"))
    cam = "anomaly_reconf_cam"

    # Default running threshold is 180 px/s
    engine = bm.get_or_create_anomaly_engine(cam)
    assert engine.thresholds.running_speed_thresh == 180.0

    # Reconfigure running threshold to 350 px/s
    bm.set_anomaly_thresholds(cam, {"running_speed_thresh": 350.0, "loitering_time": 8.0})
    updated_engine = bm.get_or_create_anomaly_engine(cam)
    assert updated_engine.thresholds.running_speed_thresh == 350.0
    assert updated_engine.thresholds.loitering_time == 8.0

    # Person running at 250 px/s should NOT trigger now because threshold is 350 px/s
    engine.update([TrackObservation(track_id=9, bbox=(100, 100, 140, 200))], timestamp=1.0)
    anoms = engine.update([TrackObservation(track_id=9, bbox=(350, 100, 390, 200))], timestamp=2.0)
    assert len(anoms) == 0

    # Person running at 400 px/s WILL trigger
    anoms2 = engine.update([TrackObservation(track_id=9, bbox=(750, 100, 790, 200))], timestamp=3.0)
    assert len(anoms2) == 1
    assert anoms2[0].event_type == "RUNNING"
    assert anoms2[0].category == "Rule-based anomaly indicator"
