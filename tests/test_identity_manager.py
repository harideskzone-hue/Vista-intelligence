import os
import uuid
import datetime
import pytest

from edge.face.types import FaceEmbedding, FaceImage, RecognitionEvent
from edge.face.identity_manager import IdentityManager

@pytest.fixture
def temp_db_path(tmp_path):
    return str(tmp_path / "test_face.db")

@pytest.fixture
def identity_manager(temp_db_path):
    return IdentityManager(temp_db_path)

def test_enroll_person(identity_manager):
    person = identity_manager.enroll_person("person_001", "John Doe", "STAFF")
    assert person.person_id == "person_001"
    assert person.display_name == "John Doe"
    assert person.category == "STAFF"
    
    # Retrieve it
    fetched = identity_manager.get_person("person_001")
    assert fetched is not None
    assert fetched.display_name == "John Doe"

def test_add_embedding(identity_manager):
    # Try adding embedding to non-existent person
    with pytest.raises(ValueError):
        identity_manager.add_embedding("person_002", b"dummy_vector", "v1", 0.9, "enrollment")
        
    # Enroll and then add
    identity_manager.enroll_person("person_002", "Jane Smith", "VISITOR")
    emb = identity_manager.add_embedding("person_002", b"dummy_vector", "v1", 0.95, "enrollment")
    assert emb.person_id == "person_002"
    
    embeddings = identity_manager.get_all_embeddings()
    assert len(embeddings) == 1
    assert embeddings[0].embedding_vector == b"dummy_vector"

def test_duplicate_image(identity_manager):
    identity_manager.enroll_person("person_003", "Alice", "STAFF")
    
    now = datetime.datetime.now(datetime.timezone.utc)
    img = FaceImage(
        image_id=str(uuid.uuid4()),
        person_id="person_003",
        path="/data/persons/person_003/1.jpg",
        camera_id="CAM-01",
        frame_id=100,
        timestamp=now,
        sharpness=45.0,
        pose_score=None,
        lighting_score=None,
        occlusion_score=None,
        recognition_similarity=0.99,
        quality_score=0.88,
        hash_sha256="abcd1234hash",
        source="LIVE"
    )
    
    assert not identity_manager.check_duplicate_image("abcd1234hash")
    identity_manager.add_face_image(img)
    assert identity_manager.check_duplicate_image("abcd1234hash")

def test_log_event(identity_manager):
    now = datetime.datetime.now(datetime.timezone.utc)
    event = RecognitionEvent(
        event_id=str(uuid.uuid4()),
        camera_id="CAM-02",
        person_id=None,
        frame_id=200,
        timestamp=now,
        similarity=None,
        quality_score=0.4,
        decision="UNKNOWN",
        image_id=None
    )
    # Should run without error
    identity_manager.log_event(event)
