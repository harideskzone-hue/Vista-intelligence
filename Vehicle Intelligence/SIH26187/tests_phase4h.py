from anpr.fuzzy_temporal_voter import FuzzyTemporalVoter

def run_test(name, obs_list, expected_text, expected_status):
    print(f"--- {name} ---")
    voter = FuzzyTemporalVoter(min_observations=3, min_confidence=0.40)
    for obs in obs_list:
        voter.add(*obs)
    
    decision = voter.decide()
    print(f"Decision : {decision.text}")
    print(f"Status   : {decision.status}")
    print(f"Confidence: {decision.confidence:.3f}")
    
    assert decision.status == expected_status, f"Expected {expected_status}, got {decision.status}"
    if expected_text is not None:
        assert decision.text == expected_text, f"Expected {expected_text}, got {decision.text}"
    else:
        assert decision.text is None, f"Expected None, got {decision.text}"
    print("PASS\n")

print("=== PHASE 4H UNIT TESTS ===")

# Test A: noisy same plate
# TN09AB1234, TN09AB1234, TN09A81234, TNO9AB1234, TN09AB1234
run_test("Test A: Noisy Same Plate", [
    ("TN09AB1234", 0.90, 0.90, 0.90, 1),
    ("TN09AB1234", 0.90, 0.90, 0.90, 2),
    ("TN09A81234", 0.85, 0.90, 0.80, 3), # 8 instead of B
    ("TNO9AB1234", 0.80, 0.90, 0.80, 4), # O instead of 0
    ("TN09AB1234", 0.90, 0.90, 0.90, 5),
], "TN09AB1234", "CONFIRMED")

# Test B: insufficient evidence to fill missing chars
# WBX, 166WBX, M66WBX
run_test("Test B: Insufficient Evidence", [
    ("WBX", 0.70, 0.80, 0.80, 1),
    ("166WBX", 0.60, 0.80, 0.70, 2),
    ("M66WBX", 0.50, 0.80, 0.70, 3),
], None, "UNKNOWN")

# Test C: garbage
run_test("Test C: Garbage", [
    ("H", 0.90, 0.90, 0.90, 1),
    ("I", 0.90, 0.90, 0.90, 2),
    ("ZXL", 0.90, 0.90, 0.90, 3),
    ("03334441231", 0.90, 0.90, 0.90, 4),
], None, "UNKNOWN")

# Test D: conflicting plates
run_test("Test D: Conflicting Plates", [
    ("TN09AB1234", 0.90, 0.90, 0.90, 1),
    ("KA01MN5678", 0.90, 0.90, 0.90, 2),
    ("TN09AB1234", 0.90, 0.90, 0.90, 3),
    ("KA01MN5678", 0.90, 0.90, 0.90, 4),
], None, "UNKNOWN")

# Test E: one excellent observation vs many bad observations
run_test("Test E: One Excellent vs Many Bad", [
    ("TN09AB1234", 0.95, 0.95, 0.95, 1),
    ("TN09A", 0.30, 0.50, 0.30, 2),
    ("TN09A", 0.30, 0.50, 0.30, 3),
    ("TN09A", 0.30, 0.50, 0.30, 4),
], None, "UNKNOWN")

print("ALL STATUS: PASS")
