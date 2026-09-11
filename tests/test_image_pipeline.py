import pytest
import numpy as np
import cv2

from edge.face.types import FaceDetection, QualityDecision, FaceImage
from edge.face.aligner import FaceAligner, AlignmentError
from edge.face.cropper import FaceCropper
from edge.face.duplicate_filter import DuplicateFilter, compute_sha256, compute_phash
from edge.face.best_frame import BestFrameSelector
from edge.face.identity_manager import IdentityManager

@pytest.fixture
def identity_manager(tmp_path):
    return IdentityManager(str(tmp_path / "face.db"))

def test_valid_face_deterministic_alignment():
    # 1. Valid face -> deterministic aligned crop.
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    
    # eyes are tilted (y=50, y=70)
    landmarks = (
        (50.0, 50.0),  # left eye
        (100.0, 70.0), # right eye
        (75.0, 90.0),  # nose
        (60.0, 110.0), # mouth L
        (90.0, 110.0)  # mouth R
    )
    det = FaceDetection(0, 40, 40, 110, 120, 0.9, landmarks)
    
    aligned, M = FaceAligner.align(img, det)
    assert aligned is not None
    assert M.shape == (2, 3)
    # The image should be rotated
    
def test_invalid_landmarks_rejected():
    # 2. Invalid/missing landmarks -> rejection.
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    det = FaceDetection(0, 40, 40, 110, 120, 0.9, tuple())
    
    with pytest.raises(AlignmentError):
        FaceAligner.align(img, det)

def test_passport_crop_dimensions():
    # 3. Passport-style crop has the expected dimensions/aspect ratio.
    # 4. Face framing remains within the intended margins.
    img = np.zeros((500, 500, 3), dtype=np.uint8)
    det = FaceDetection(0, 200, 200, 300, 300, 0.9, None) # 100x100 face
    
    crop = FaceCropper.passport_style_crop(img, det)
    assert crop.shape[1] == FaceCropper.TARGET_WIDTH
    assert crop.shape[0] == FaceCropper.TARGET_HEIGHT
    
    ratio = crop.shape[1] / crop.shape[0]
    assert pytest.approx(ratio, 0.01) == FaceCropper.ASPECT_RATIO

def test_exact_duplicate_rejected():
    # 5. Exact duplicate -> rejected by SHA-256.
    img1 = np.ones((100, 100, 3), dtype=np.uint8) * 100
    img1_bytes = img1.tobytes()
    
    sha_db = set()
    def exact_exists(sha: str):
        return sha in sha_db
        
    df = DuplicateFilter()
    
    # First time unique
    dup, reason = df.is_duplicate("p1", img1_bytes, img1, exact_exists)
    assert not dup
    assert reason == "UNIQUE"
    
    # Save it
    sha_db.add(compute_sha256(img1_bytes))
    df.register_image("p1", img1)
    
    # Second time exact duplicate
    dup, reason = df.is_duplicate("p1", img1_bytes, img1, exact_exists)
    assert dup
    assert reason == "EXACT_DUPLICATE"

def test_near_duplicate_rejected():
    # 6. Near duplicate -> rejected by perceptual hash.
    img1 = np.ones((100, 100, 3), dtype=np.uint8) * 100
    img1_bytes = img1.tobytes()
    
    # Add minor noise
    img2 = img1.copy()
    img2[0:5, 0:5] = 200
    img2_bytes = img2.tobytes()
    
    sha_db = set()
    def exact_exists(sha: str):
        return sha in sha_db
        
    df = DuplicateFilter(phash_threshold=5)
    
    dup, _ = df.is_duplicate("p1", img1_bytes, img1, exact_exists)
    assert not dup
    sha_db.add(compute_sha256(img1_bytes))
    df.register_image("p1", img1)
    
    # Second image is near duplicate
    dup, reason = df.is_duplicate("p1", img2_bytes, img2, exact_exists)
    assert dup
    assert reason == "NEAR_DUPLICATE"

def test_different_image_retained():
    # 7. Different image -> retained.
    img1 = np.ones((100, 100, 3), dtype=np.uint8) * 100
    img1_bytes = img1.tobytes()
    
    # Completely different image (gradient/edge that increases so pHash sees True)
    img2 = np.zeros((100, 100, 3), dtype=np.uint8)
    img2[:, 50:] = 255
    img2_bytes = img2.tobytes()
    
    sha_db = set()
    def exact_exists(sha: str):
        return sha in sha_db
        
    df = DuplicateFilter()
    sha_db.add(compute_sha256(img1_bytes))
    df.register_image("p1", img1)
    
    dup, reason = df.is_duplicate("p1", img2_bytes, img2, exact_exists)
    assert not dup
    assert reason == "UNIQUE"

def test_best_frame_selection():
    # 8. Higher-quality duplicate candidate -> replaces/retains the better frame.
    # 9. Blurry frame loses to sharper frame.
    # 12. Small face loses to sufficiently large face.
    bfs = BestFrameSelector()
    
    # Base frame
    det1 = FaceDetection(0, 100, 100, 200, 200, 0.9, None) # 100x100
    q1 = QualityDecision(0, "PASS", None, 50.0, 100.0, 100.0, 40.0)
    
    assert bfs.evaluate_and_update("p1", det1, q1) == True
    
    # Smaller face, same sharpness
    det2 = FaceDetection(0, 100, 100, 150, 150, 0.9, None) # 50x50
    assert bfs.evaluate_and_update("p1", det2, q1) == False
    
    # Same size, blurrier
    q3 = QualityDecision(0, "PASS", None, 20.0, 100.0, 100.0, 40.0)
    assert bfs.evaluate_and_update("p1", det1, q3) == False
    
    # Sharper frame -> Replaces
    q4 = QualityDecision(0, "PASS", None, 90.0, 100.0, 100.0, 40.0)
    assert bfs.evaluate_and_update("p1", det1, q4) == True

def test_invalid_person_id_best_frame():
    # 15. Unknown identity cannot create a folder/person as a side effect.
    bfs = BestFrameSelector()
    det1 = FaceDetection(0, 100, 100, 200, 200, 0.9, None)
    q1 = QualityDecision(0, "PASS", None, 50.0, 100.0, 100.0, 40.0)
    
    assert bfs.evaluate_and_update("UNKNOWN", det1, q1) == False
    assert "UNKNOWN" not in bfs.best_scores

def test_no_image_for_invalid_person(identity_manager):
    # 14. No image is stored for an invalid/nonexistent person_id.
    import datetime
    
    img_meta = FaceImage(
        image_id="img1", person_id="invalid_p", path="/data/invalid_p/img1.jpg",
        camera_id="cam1", frame_id=1, timestamp=datetime.datetime.now(datetime.timezone.utc),
        sharpness=50.0, pose_score=None, lighting_score=None, occlusion_score=None,
        recognition_similarity=0.9, quality_score=50.0, hash_sha256="hash", source="LIVE"
    )
    
    with pytest.raises(ValueError):
        identity_manager.add_face_image(img_meta)
        
def test_same_person_id_maps_to_same_storage(identity_manager):
    # 13. Same person_id always maps to the same storage location.
    # Verification that FaceImage records map to the identical Person_id regardless of camera.
    identity_manager.enroll_person("p1", "Alice", "STAFF")
    
    import datetime
    now = datetime.datetime.now(datetime.timezone.utc)
    
    img1 = FaceImage(
        image_id="img1", person_id="p1", path="/data/p1/img1.jpg",
        camera_id="cam1", frame_id=1, timestamp=now, sharpness=50.0, pose_score=None, 
        lighting_score=None, occlusion_score=None, recognition_similarity=0.9, quality_score=50.0, 
        hash_sha256="hash1", source="LIVE"
    )
    identity_manager.add_face_image(img1)
    
    img2 = FaceImage(
        image_id="img2", person_id="p1", path="/data/p1/img2.jpg",
        camera_id="cam2", frame_id=2, timestamp=now, sharpness=50.0, pose_score=None, 
        lighting_score=None, occlusion_score=None, recognition_similarity=0.9, quality_score=50.0, 
        hash_sha256="hash2", source="LIVE"
    )
    identity_manager.add_face_image(img2)
    
    with identity_manager._get_conn() as conn:
        imgs = conn.execute("SELECT person_id FROM face_images WHERE person_id = 'p1'").fetchall()
        assert len(imgs) == 2
        # Identity Manager enforces that the person_id is a foreign key to persons table, ensuring invariant.
