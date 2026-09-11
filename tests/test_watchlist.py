import os
import json
import pytest
import numpy as np

from edge.face.identity_manager import IdentityManager
from edge.face.watchlist import Watchlist, WatchlistIntegrityError

@pytest.fixture
def identity_manager(tmp_path):
    db_path = str(tmp_path / "face.db")
    return IdentityManager(db_path)

@pytest.fixture
def index_dir(tmp_path):
    return str(tmp_path / "faiss_index")

def test_empty_watchlist(identity_manager, index_dir):
    wl = Watchlist(identity_manager, index_dir, dimension=512)
    assert wl.index.ntotal == 0
    # Safe search on empty
    query = np.zeros(512, dtype=np.float32)
    results = wl.search(query, top_k=5)
    assert len(results) == 0

def test_build_and_search_multiple_embeddings(identity_manager, index_dir):
    # Enroll Person
    identity_manager.enroll_person("person_1", "Alice", "STAFF")
    
    # 2 embeddings for Person 1
    v1 = np.zeros(512, dtype=np.float32)
    v1[0] = 1.0  # normalized
    
    v2 = np.zeros(512, dtype=np.float32)
    v2[1] = 1.0  # normalized
    
    emb1 = identity_manager.add_embedding("person_1", v1.tobytes(), "v1", 0.9, "enr")
    emb2 = identity_manager.add_embedding("person_1", v2.tobytes(), "v1", 0.9, "enr")
    
    wl = Watchlist(identity_manager, index_dir, dimension=512)
    
    assert wl.index.ntotal == 2
    
    # Search closest to v1
    results = wl.search(v1, top_k=5)
    assert len(results) == 2
    
    # Check top match
    assert results[0][0] == "person_1"
    assert results[0][1] == emb1.embedding_id
    assert results[0][2] > 0.99  # similarity approx 1.0
    
    # Check second match
    assert results[1][0] == "person_1"
    assert results[1][1] == emb2.embedding_id
    assert results[1][2] < 0.1   # orthogonal

def test_integrity_db_missing_person(identity_manager, index_dir):
    identity_manager.enroll_person("person_1", "Alice", "STAFF")
    v1 = np.zeros(512, dtype=np.float32)
    identity_manager.add_embedding("person_1", v1.tobytes(), "v1", 0.9, "enr")
    
    wl = Watchlist(identity_manager, index_dir, dimension=512)
    
    # Manually delete person from DB (simulate corruption)
    with identity_manager._get_conn() as conn:
        conn.execute("DELETE FROM persons WHERE person_id = 'person_1'")
        conn.commit()
    
    # Reload should detect inconsistency and fail closed or rebuild.
    # Rebuilding will also fail because embedding points to missing person.
    with pytest.raises(WatchlistIntegrityError):
        Watchlist(identity_manager, index_dir, dimension=512)

def test_integrity_db_embedding_missing_from_faiss(identity_manager, index_dir):
    identity_manager.enroll_person("person_1", "Alice", "STAFF")
    v1 = np.zeros(512, dtype=np.float32)
    identity_manager.add_embedding("person_1", v1.tobytes(), "v1", 0.9, "enr")
    
    wl = Watchlist(identity_manager, index_dir, dimension=512)
    
    # Now add to DB directly without updating FAISS
    v2 = np.ones(512, dtype=np.float32)
    identity_manager.add_embedding("person_1", v2.tobytes(), "v1", 0.9, "enr")
    
    # Re-instantiating will detect mismatch and rebuild index safely
    wl2 = Watchlist(identity_manager, index_dir, dimension=512)
    assert wl2.index.ntotal == 2

def test_integrity_corrupt_index_does_not_silently_match(identity_manager, index_dir):
    identity_manager.enroll_person("person_1", "Alice", "STAFF")
    v1 = np.zeros(512, dtype=np.float32)
    emb1 = identity_manager.add_embedding("person_1", v1.tobytes(), "v1", 0.9, "enr")
    
    wl = Watchlist(identity_manager, index_dir, dimension=512)
    
    # Manually corrupt the mapping json
    with open(wl.mapping_path, 'r') as f:
        mapping = json.load(f)
    mapping["0"] = "invalid_emb_id"
    with open(wl.mapping_path, 'w') as f:
        json.dump(mapping, f)
        
    # Loading will fail validation and it will rebuild correctly
    wl2 = Watchlist(identity_manager, index_dir, dimension=512)
    assert wl2.index.ntotal == 1
    # Check it rebuilt to correct mapping
    results = wl2.search(v1, top_k=1)
    assert results[0][1] == emb1.embedding_id

def test_search_fetches_from_db_to_validate(identity_manager, index_dir):
    identity_manager.enroll_person("person_1", "Alice", "STAFF")
    v1 = np.zeros(512, dtype=np.float32)
    identity_manager.add_embedding("person_1", v1.tobytes(), "v1", 0.9, "enr")
    
    wl = Watchlist(identity_manager, index_dir, dimension=512)
    
    # Corrupt DB after index is loaded in memory
    with identity_manager._get_conn() as conn:
        conn.execute("DELETE FROM face_embeddings")
        conn.commit()
        
    # Search should fail closed when resolving the embedding
    with pytest.raises(WatchlistIntegrityError):
        wl.search(v1, top_k=1)
