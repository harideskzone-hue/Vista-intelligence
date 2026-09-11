import pytest
import os
import time
from unittest.mock import patch, MagicMock
from face_api.app.core.storage import Storage

@pytest.fixture
def storage(tmp_path):
    with patch("face_api.app.core.storage.SQLITE_PATH", str(tmp_path / "test.db")), \
         patch("face_api.app.core.storage.IMAGES_DIR", str(tmp_path / "images")), \
         patch("face_api.app.core.storage.RECORDINGS_DIR", str(tmp_path / "recordings")):
        s = Storage()
        yield s

def test_recording_metadata_lifecycle(storage):
    # Start recording
    rec_id = storage.start_recording("cam_1", "Main Gate", "cam_1_abc.mp4")
    assert rec_id is not None
    
    # Retrieve
    rec = storage.get_recording(rec_id)
    assert rec['camera_id'] == "cam_1"
    assert rec['status'] == "RECORDING"
    assert rec['duration'] == 0.0
    
    # Finish recording
    storage.finish_recording(rec_id, end_time="2026-01-01T10:00:00", duration=45.5, size=1024)
    
    rec_finished = storage.get_recording(rec_id)
    assert rec_finished['status'] == "COMPLETED"
    assert rec_finished['duration'] == 45.5
    assert rec_finished['size'] == 1024

def test_recording_deletion(storage, tmp_path):
    rec_id = storage.start_recording("cam_2", "Side Gate", "cam_2_def.mp4")
    storage.finish_recording(rec_id, "2026-01-01T10:01:00", 30.0, 500)
    
    # Ensure it exists
    assert storage.get_recording(rec_id) is not None
    
    # Delete
    storage.delete_recording(rec_id)
    assert storage.get_recording(rec_id) is None
    
def test_list_recordings(storage):
    for i in range(3):
        storage.start_recording(f"cam_{i}", f"Gate {i}", f"cam_{i}.mp4")
        
    records = storage.get_recordings()
    assert len(records) == 3
