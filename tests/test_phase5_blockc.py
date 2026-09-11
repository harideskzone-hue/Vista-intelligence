import os
import pytest
import numpy as np
import uuid
import cv2

from edge.face.types import FaceImage, Person, FaceEmbedding
from edge.face.identity_manager import IdentityManager
import face_api.app.services.face_service as fs_module
from face_api.app.services.face_service import FaceService, _get_edge_components

@pytest.fixture(autouse=True)
def mocked_encoder(monkeypatch):
    class MockEncoder:
        def detect_and_encode(self, image):
            emb = np.random.rand(512).astype(np.float32)
            emb = emb / np.linalg.norm(emb)
            info = {'det_score': 0.99, 'bbox': [0,0,100,100]}
            return emb, info

    monkeypatch.setattr(fs_module, "get_encoder", lambda: MockEncoder())

@pytest.fixture
def clean_service(tmp_path):
    old_val = os.environ.get("SIH26187_DATA")
    os.environ["SIH26187_DATA"] = str(tmp_path)
    
    fs_module._identity_manager = None
    fs_module._watchlist = None
    fs_module._temporal_matcher = None
    fs_module._face_matcher = None
    fs_module._best_frame_selector = None
    fs_module._duplicate_filter = None
    
    yield FaceService()
    
    if old_val:
        os.environ["SIH26187_DATA"] = old_val
    else:
        del os.environ["SIH26187_DATA"]

def create_mock_jpeg():
    img = np.zeros((112,112,3), dtype=np.uint8)
    _, buffer = cv2.imencode('.jpg', img)
    return buffer.tobytes()

def test_task12_reference_images(clean_service):
    # Enroll a person
    img_bytes = create_mock_jpeg()
    res = clean_service.enroll_person("Ref Person", "STAFF", img_bytes)
    assert res['success']
    person_id = res['person_id']
    
    im, wl, _, _, _, _ = _get_edge_components()
    
    # Check reference images
    ref_images = im.get_reference_images(person_id)
    assert len(ref_images) == 1
    
    # Add reference image
    add_res = clean_service.add_reference_images_to_person(person_id, [create_mock_jpeg(), create_mock_jpeg()])
    assert add_res['success']
    assert add_res['added_count'] == 2
    
    ref_images_after = im.get_reference_images(person_id)
    assert len(ref_images_after) == 3
    
    assert wl.index.ntotal == 3
    
    # Delete one reference image
    del_res = clean_service.delete_reference_image(ref_images_after[0])
    assert del_res['success']
    
    # Check that FAISS and DB are synced
    ref_images_final = im.get_reference_images(person_id)
    assert len(ref_images_final) == 2
    assert wl.index.ntotal == 2

def test_task13_classification(clean_service):
    res = clean_service.enroll_person("Class Person", "STAFF", create_mock_jpeg())
    p_id = res['person_id']
    
    # Update to wanted
    update_res = clean_service.update_person_edge(p_id, {"classification": "wanted"})
    assert update_res['success']
    
    im, wl, _, _, _, _ = _get_edge_components()
    p = im.get_person(p_id)
    assert p.classification == "wanted"
    
    # Person should still be searchable
    assert wl.index.ntotal == 1

def test_task14_disabled_status(clean_service):
    res = clean_service.enroll_person("Active Person", "STAFF", create_mock_jpeg())
    p_id = res['person_id']
    im, wl, _, _, _, _ = _get_edge_components()
    
    assert wl.index.ntotal == 1
    
    # Disable person
    clean_service.update_person_edge(p_id, {"status": "DISABLED"})
    
    # Verify removed from FAISS
    assert wl.index.ntotal == 0
    
    # Enable person
    clean_service.update_person_edge(p_id, {"status": "ACTIVE"})
    
    # Verify added back to FAISS
    assert wl.index.ntotal == 1

def test_task15_sync_and_delete_person(clean_service):
    res = clean_service.enroll_person("Delete Me", "STAFF", create_mock_jpeg())
    p_id = res['person_id']
    
    im, wl, _, _, _, _ = _get_edge_components()
    assert wl.index.ntotal == 1
    
    del_res = clean_service.delete_person(p_id)
    assert del_res['success']
    
    assert im.get_person(p_id) is None
    assert wl.index.ntotal == 0
    assert len(im.get_reference_images(p_id)) == 0

def test_task16_evidence_separation(clean_service):
    import datetime
    now = datetime.datetime.now(datetime.timezone.utc)
    im, wl, _, _, _, _ = _get_edge_components()
    
    # Simulate an evidence image
    p_id = "test_person"
    im.execute_enrollment_transaction(
        Person(p_id, "Test", "STAFF", "normal", now, "ACTIVE"),
        FaceEmbedding("e1", p_id, b"123", "v1", now, 1.0, "ENROLLMENT"),
        FaceImage("i1", p_id, "/path", "C1", 0, now, 1.0, None, None, None, 1.0, 1.0, "h1", "ENROLLMENT", "e1")
    )
    
    im.add_face_image(FaceImage("i2", p_id, "/path2", "C1", 0, now, 1.0, None, None, None, 1.0, 1.0, "h2", "RECOGNITION", None))
    
    # Delete reference image
    im.delete_reference_image("i1")
    
    # Evidence should still exist
    with im._get_conn() as conn:
        evidence = conn.execute("SELECT * FROM face_images WHERE image_id = 'i2'").fetchone()
        assert evidence is not None
        
    # Trying to delete evidence as reference should raise ValueError
    with pytest.raises(ValueError):
        im.delete_reference_image("i2")
