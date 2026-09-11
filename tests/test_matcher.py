import pytest
import numpy as np

from edge.face.types import FaceMatchResult
from edge.face.watchlist import Watchlist, WatchlistIntegrityError
from edge.face.temporal_matcher import TemporalMatcher
from edge.face.matcher import FaceMatcher
from edge.face.identity_manager import IdentityManager


@pytest.fixture
def identity_manager(tmp_path):
    return IdentityManager(str(tmp_path / "face.db"))


@pytest.fixture
def watchlist(identity_manager, tmp_path):
    return Watchlist(identity_manager, str(tmp_path / "index"), dimension=2)


@pytest.fixture
def populated_matcher(identity_manager, watchlist):
    # Enroll Person 1
    identity_manager.enroll_person("person_1", "Alice", "STAFF")
    v1 = np.array([1.0, 0.0], dtype=np.float32)
    identity_manager.add_embedding("person_1", v1.tobytes(), "v1", 0.9, "enr")
    
    # Enroll Person 2
    identity_manager.enroll_person("person_2", "Bob", "STAFF")
    v2 = np.array([0.0, 1.0], dtype=np.float32)
    identity_manager.add_embedding("person_2", v2.tobytes(), "v1", 0.9, "enr")
    
    watchlist.rebuild_index()
    
    # 3 frames to confirm, 5 frames to switch
    tm = TemporalMatcher(confirmation_frames=3, switch_frames=5)
    
    # Threshold 0.5, Margin 0.1
    return FaceMatcher(watchlist, tm, operational_threshold=0.5, margin_threshold=0.1)


def test_empty_watchlist(identity_manager, tmp_path):
    wl = Watchlist(identity_manager, str(tmp_path / "empty_idx"), dimension=2)
    tm = TemporalMatcher()
    matcher = FaceMatcher(wl, tm)
    
    # 11. Empty watchlist behaves safely -> NO_WATCHLIST
    res = matcher.match("cam1", "t1", np.array([1.0, 0.0], dtype=np.float32))
    assert res.match_status == "NO_WATCHLIST"
    assert res.matched_identity_id is None


def test_poor_quality_rejected(populated_matcher):
    # 14. Poor-quality input cannot produce a confident match.
    emb = np.array([1.0, 0.0], dtype=np.float32)
    res = populated_matcher.match("cam1", "t1", emb, quality_pass=False)
    assert res.match_status == "UNKNOWN"


def test_below_threshold(populated_matcher):
    # 2. Below threshold -> UNKNOWN
    # Vector orthogonal or small magnitude yielding low similarity
    emb = np.array([-1.0, -1.0], dtype=np.float32) # Dot product with [1,0] and [0,1] is -1.0
    
    for _ in range(4): # Enough frames for temporal confirm if it were to match
        res = populated_matcher.match("cam1", "t1", emb)
        
    assert res.match_status == "UNKNOWN"
    assert res.matched_identity_id is None


def test_insufficient_margin(populated_matcher):
    # 3. Strong top-1 but insufficient margin -> UNKNOWN (AMBIGUOUS)
    # v1 = [1,0], v2 = [0,1]. We give [0.707, 0.707] (approx equal distance)
    # similarity to both is 0.707. Margin is 0.0.
    emb = np.array([0.707, 0.707], dtype=np.float32)
    
    for _ in range(4):
        res = populated_matcher.match("cam1", "t1", emb)
        
    # Temporal matcher won't confirm because candidate_id is None from matcher
    assert res.match_status == "UNKNOWN"
    assert res.matched_identity_id is None


def test_strong_top1_match_confirmation(populated_matcher):
    # 1. Strong top-1 match -> MATCH
    # 4. Correct top-1 with adequate margin -> candidate accepted
    # 5. Multiple consecutive agreeing observations -> temporal confirmation.
    emb = np.array([1.0, 0.0], dtype=np.float32)
    
    # Frame 1: Candidate accepted, but temporally unconfirmed
    res1 = populated_matcher.match("cam1", "t1", emb)
    assert res1.match_status == "AMBIGUOUS"
    assert res1.matched_identity_id is None
    
    # Frame 2: Still unconfirmed
    res2 = populated_matcher.match("cam1", "t1", emb)
    assert res2.match_status == "AMBIGUOUS"
    
    # Frame 3: Confirmed (confirmation_frames = 3)
    res3 = populated_matcher.match("cam1", "t1", emb)
    assert res3.match_status == "MATCH"
    assert res3.matched_identity_id == "person_1"


def test_inconsistent_observations(populated_matcher):
    # 6. Inconsistent observations -> no confirmation
    e1 = np.array([1.0, 0.0], dtype=np.float32) # Person 1
    e2 = np.array([0.0, 1.0], dtype=np.float32) # Person 2
    
    populated_matcher.match("cam1", "t1", e1)
    populated_matcher.match("cam1", "t1", e2)
    res = populated_matcher.match("cam1", "t1", e1)
    
    assert res.match_status == "AMBIGUOUS"
    assert res.matched_identity_id is None


def test_identity_switch_guard(populated_matcher):
    # 7. Person 1 -> Person 2 oscillation -> identity-switch guard prevents false reassignment.
    e1 = np.array([1.0, 0.0], dtype=np.float32)
    e2 = np.array([0.0, 1.0], dtype=np.float32)
    
    # Confirm Person 1
    for _ in range(3):
        populated_matcher.match("cam1", "t1", e1)
        
    # Now send Person 2 frames. It takes 5 frames to switch.
    for i in range(4):
        res = populated_matcher.match("cam1", "t1", e2)
        # Should still be Person 1 (guarded)
        assert res.match_status == "MATCH"
        assert res.matched_identity_id == "person_1"
        
    # 5th frame of Person 2 switches identity
    res = populated_matcher.match("cam1", "t1", e2)
    assert res.match_status == "MATCH"
    assert res.matched_identity_id == "person_2"


def test_independent_cameras(populated_matcher):
    # 8. Different cameras maintain independent temporal state.
    # 9. Same person across cameras resolves to the same persistent person_id.
    e1 = np.array([1.0, 0.0], dtype=np.float32)
    
    populated_matcher.match("cam1", "t1", e1)
    populated_matcher.match("cam1", "t1", e1)
    
    # Cam2 starts new
    res_cam2 = populated_matcher.match("cam2", "t1", e1)
    assert res_cam2.match_status == "AMBIGUOUS" # requires 3 frames on cam2
    
    # Cam1 confirms
    res_cam1 = populated_matcher.match("cam1", "t1", e1)
    assert res_cam1.match_status == "MATCH"
    assert res_cam1.matched_identity_id == "person_1"
    
    # Cam2 continues and confirms same ID
    populated_matcher.match("cam2", "t1", e1)
    res_cam2 = populated_matcher.match("cam2", "t1", e1)
    assert res_cam2.match_status == "MATCH"
    assert res_cam2.matched_identity_id == "person_1"


def test_restart_clears_state(populated_matcher):
    # 10. Restart clears temporal state but does not alter persistent identity.
    e1 = np.array([1.0, 0.0], dtype=np.float32)
    
    # Confirm
    for _ in range(3):
        populated_matcher.match("cam1", "t1", e1)
        
    # Simulate restart by clearing temporal matcher
    populated_matcher.temporal.clear_all()
    
    # Next frame is unconfirmed
    res = populated_matcher.match("cam1", "t1", e1)
    assert res.match_status == "AMBIGUOUS"
    assert res.matched_identity_id is None
    
    # Still resolves to same persistent identity after confirmation
    populated_matcher.match("cam1", "t1", e1)
    res = populated_matcher.match("cam1", "t1", e1)
    assert res.match_status == "MATCH"
    assert res.matched_identity_id == "person_1"


def test_integrity_failure_fails_closed(populated_matcher):
    # 12. Watchlist integrity failure -> fail closed
    # Corrupt the index mappings
    populated_matcher.watchlist.faiss_to_emb[0] = "invalid_id"
    
    e1 = np.array([1.0, 0.0], dtype=np.float32)
    res = populated_matcher.match("cam1", "t1", e1)
    assert res.match_status == "UNKNOWN"


def test_unknown_never_enrolls(populated_matcher):
    # 13. Unknown face never causes enrollment.
    e_unknown = np.array([-1.0, -1.0], dtype=np.float32)
    for _ in range(10):
        populated_matcher.match("cam1", "t1", e_unknown)
        
    # DB still has 2 persons
    assert len(populated_matcher.watchlist.identity_manager.get_all_persons()) == 2
