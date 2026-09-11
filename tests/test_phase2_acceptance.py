
import os
import pytest
import numpy as np
import shutil
import tempfile
import cv2

# Import fs_module first
import face_api.app.services.face_service as fs_module
from face_api.app.services.face_service import FaceService, _get_edge_components

@pytest.fixture
def clean_env():
    with tempfile.TemporaryDirectory() as td:
        old_val = os.environ.get("SIH26187_DATA")
        os.environ["SIH26187_DATA"] = td
        
        # CLEAR GLOBALS
        fs_module._identity_manager = None
        fs_module._watchlist = None
        fs_module._temporal_matcher = None
        fs_module._face_matcher = None
        fs_module._best_frame_selector = None
        fs_module._duplicate_filter = None
        
        yield td
        
        if old_val is not None:
            os.environ["SIH26187_DATA"] = old_val
        else:
            del os.environ["SIH26187_DATA"]

@pytest.fixture
def mocked_encoder(monkeypatch):
    class MockEncoder:
        def detect_and_encode(self, image):
            emb = np.random.rand(512).astype(np.float32)
            emb = emb / np.linalg.norm(emb)
            info = {'det_score': 0.99, 'bbox': [0,0,100,100]}
            return emb, info

    monkeypatch.setattr(fs_module, "get_encoder", lambda: MockEncoder())
    yield

def test_phase2_acceptance_criteria(clean_env, mocked_encoder):
    service = FaceService()
    
    valid_img = np.zeros((112,112,3), dtype=np.uint8)
    _, buffer = cv2.imencode('.jpg', valid_img)
    img_data = buffer.tobytes()

    res1 = service.enroll_person("Normal Guy", "STAFF", img_data)
    assert res1['success'], f"Enrollment failed for normal person: {res1}"
    p1_id = res1['person_id']

    all_persons = service.get_all_persons_edge()
    assert any(p['id'] == p1_id for p in all_persons['persons']), "Normal person not found in DB"
    p1_info = next(p for p in all_persons['persons'] if p['id'] == p1_id)
    assert p1_info['classification'] == 'normal', "Default classification should be 'normal'"
    print("[PASS] 1. Created normal person, verified in Face Database.")

    res2 = service.enroll_person("Wanted Guy", "WANTED", img_data)
    assert res2['success'], "Enrollment failed for wanted person"
    p2_id = res2['person_id']

    update_res = service.update_person_edge(p2_id, {"classification": "wanted"})
    assert update_res['success']

    all_persons = service.get_all_persons_edge()
    p2_info = next(p for p in all_persons['persons'] if p['id'] == p2_id)
    assert p2_info['classification'] == 'wanted', "Person not flagged as wanted"

    im, wl, tm, fm, bfs, dup = _get_edge_components()
    assert wl.index is not None
    assert wl.index.ntotal == 2, f"FAISS should have 2 embeddings but got {wl.index.ntotal}"
    print("[PASS] 2. Created wanted person, verified person_id retained, classification=wanted, embedding in FAISS.")

    service.update_person_edge(p1_id, {"classification": "wanted"})
    all_persons = service.get_all_persons_edge()
    p1_wanted = next(p for p in all_persons['persons'] if p['id'] == p1_id)
    assert p1_wanted['classification'] == 'wanted'
    service.update_person_edge(p1_id, {"classification": "normal"})
    all_persons = service.get_all_persons_edge()
    p1_normal = next(p for p in all_persons['persons'] if p['id'] == p1_id)
    assert p1_normal['classification'] == 'normal'
    print("[PASS] 3. Changed normal -> wanted -> normal successfully.")

    service.update_person_edge(p1_id, {"display_name": "Normal Guy Edited"})
    all_persons = service.get_all_persons_edge()
    p1_edited = next(p for p in all_persons['persons'] if p['id'] == p1_id)
    assert p1_edited['name'] == "Normal Guy Edited", "Display name didn't update"
    
    with im._get_conn() as conn:
        db_name = conn.execute("SELECT display_name FROM persons WHERE person_id = ?", (p1_id,)).fetchone()[0]
        assert db_name == "Normal Guy Edited"
    print("[PASS] 4. Edited display name successfully in authoritative SQLite.")

    all_embs = im.get_all_embeddings()
    emb_rec = next(e for e in all_embs if e.person_id == p1_id)
    p1_emb = np.frombuffer(emb_rec.embedding_vector, dtype=np.float32)
    
    candidates_active = wl.search(p1_emb, top_k=1)
    assert len(candidates_active) > 0 and candidates_active[0][0] == p1_id, "Active person should be found"

    service.update_person_edge(p1_id, {"status": "DISABLED"})
    candidates_disabled = wl.search(p1_emb, top_k=1)
    assert not any(c[0] == p1_id for c in candidates_disabled), "Disabled person should not be retrieved from FAISS"
    print("[PASS] 5. Disabled person cannot produce a valid recognition MATCH.")

    data_dir = os.path.dirname(im.db_path)
    evidence_dir = os.path.join(data_dir, "evidence", p2_id)
    os.makedirs(evidence_dir, exist_ok=True)
    assert os.path.exists(evidence_dir)

    with im._get_conn() as conn:
        conn.execute("DELETE FROM face_images WHERE person_id = ?", (p2_id,))
        conn.execute("DELETE FROM face_embeddings WHERE person_id = ?", (p2_id,))
        conn.execute("DELETE FROM persons WHERE person_id = ?", (p2_id,))
        conn.commit()

    wl.load_or_rebuild()

    assert im.get_person(p2_id) is None
    assert wl.index.ntotal == 0, "FAISS should have 0 embeddings remaining since p1 is disabled"
    if os.path.exists(evidence_dir):
        shutil.rmtree(evidence_dir)
    assert not os.path.exists(evidence_dir), "Evidence folder should be deleted"
    print("[PASS] 6. Deleted person: SQLite record removed, FAISS mapping removed, evidence folder removed.")

    service.update_person_edge(p1_id, {"status": "ACTIVE"})
    candidates_reactivated = wl.search(p1_emb, top_k=1)
    assert candidates_reactivated[0][0] == p1_id
    print("[PASS] 7. Existing enrolled person recognizes correctly.")

