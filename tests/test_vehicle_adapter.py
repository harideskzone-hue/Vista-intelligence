import pytest
import os
import sys
import numpy as np
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath('.'))
from face_api.run import app
from edge.vehicle.adapter import VehicleIntelligenceAdapter
from edge.vehicle.schemas import VehicleObservation
from edge.vehicle.manager import get_vehicle_manager

client = TestClient(app)

def test_vehicle_pipeline_import_and_instantiation():
    adapter = VehicleIntelligenceAdapter(camera_id="TEST_01")
    assert adapter.pipeline is not None
    assert adapter.config.vehicle_model_path.endswith("yolo11n.pt")

def test_vehicle_observation_schema_and_adapter_normalization():
    adapter = VehicleIntelligenceAdapter(camera_id="TEST_01")
    from core.schemas import VehicleRecord, VehicleClass, ANPRResult, PlateStatus
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    
    frame[100:200, 100:200] = [0, 0, 255]
    
    rec = VehicleRecord(
        track_id=1,
        vehicle_class=VehicleClass.CAR,
        vehicle_confidence=0.9,
        bbox=[100, 100, 200, 200],
        anpr=ANPRResult(plate="DL8CAF5030", status=PlateStatus.CONFIRMED, confidence=0.85, format_valid=True),
        first_seen=1.0,
        last_seen=1.0,
        last_frame=1
    )
    
    obs = adapter.normalize_record(rec, frame, timestamp=1.0)
    assert isinstance(obs, VehicleObservation)
    assert obs.camera_id == "TEST_01"
    assert obs.plate_text == "DL8CAF5030"
    assert obs.plate_status == "CONFIRMED"
    assert obs.color == "Red"
    assert obs.make == "UNKNOWN"
    assert obs.model == "UNKNOWN"
    
def test_camera_isolation():
    manager = get_vehicle_manager()
    a1 = manager.get_or_create_adapter("cam_north")
    a2 = manager.get_or_create_adapter("cam_south")
    
    assert a1 is not a2
    assert a1.camera_id == "cam_north"
    assert a2.camera_id == "cam_south"
    
def test_eventhub_and_clip_manager_integration():
    manager = get_vehicle_manager()
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    from core.schemas import VehicleRecord, VehicleClass, ANPRResult, PlateStatus
    rec = VehicleRecord(
        track_id=101,
        vehicle_class=VehicleClass.CAR,
        vehicle_confidence=0.9,
        bbox=[10, 10, 20, 20],
        anpr=ANPRResult(plate="UP14AA1234", status=PlateStatus.PROBABLE, confidence=0.8, format_valid=True),
        first_seen=1.0,
        last_seen=1.0,
        last_frame=1
    )
    obs = manager.get_or_create_adapter("cam_eh_test").normalize_record(rec, frame, timestamp=10.0)
    
    manager._handle_events("cam_eh_test", frame, 10.0, obs)
    
    assert 101 in manager._emitted_detection["cam_eh_test"]
    assert 101 in manager._emitted_identification["cam_eh_test"]

def test_zero_filesystem_path_leakage():
    response = client.get("/api/vehicle/events")
    assert response.status_code == 200
    events = response.json()
    for e in events:
        assert "clip_path" not in e or not str(e["clip_path"]).startswith("/")
        assert "clip_path" not in e or not str(e["clip_path"]).startswith("C:\\")
