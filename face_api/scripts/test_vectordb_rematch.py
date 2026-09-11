
import os
import sys
import argparse
import numpy as np  # type: ignore
import faiss  # type: ignore

# Add parent directory to path so we can import 'app'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.vector_db import get_vector_db  # type: ignore

def test_rematch_logic():
    print("--- Test: VectorDB Soft-Delete Rematch Logic ---")
    
    # 1. Init VectorDB
    db = get_vector_db()
    # OVERRIDE path to use a test file (crucial to avoid wiping prod DB)
    db.index_path = 'test_faiss.index'
    # Reset to be clean (this will clear test_faiss.index if it exists, and init empty in-memory index)
    db.reset()
    
    print("Initialized empty DB (Test Mode).")
    
    # 2. Create two similar vectors
    # Person A: Base vector
    vec_a = np.random.rand(512).astype(np.float32)
    vec_a /= np.linalg.norm(vec_a)
    
    # Person B: Very similar to A (e.g. 0.95 similarity)
    # We can create B by adding small noise to A
    noise = np.random.rand(512).astype(np.float32) * 0.1
    vec_b = vec_a + noise
    vec_b /= np.linalg.norm(vec_b)
    
    # Verify similarity
    sim = np.dot(vec_a, vec_b)
    print(f"Similarity between A and B: {sim:.4f}")
    
    # 3. Add to DB
    id_a = "person_a"
    id_b = "person_b"
    
    db.add_embedding(id_a, vec_a) # Index 0
    db.add_embedding(id_b, vec_b) # Index 1
    
    print("Added Person A and Person B.")
    
    # 4. Search for A (should match A first)
    print("\nSearch for A (Before Deletion):")
    results = db.search(vec_a, top_k=1)
    print("Results:", results)
    if results and results[0][0] == id_a:
        print("SUCCESS: Matched A.")
    else:
        print("FAIL: Did not match A.")
        return

    # 5. Delete A (Soft delete - removes from mapping but keeps index)
    print("\nDeleting Person A...")
    db.remove(id_a)
    
    # 6. Search for A again 
    # Logic: FAISS will still return A's vector (Index 0) as top match (1.0 similarity)
    # But filtering should skip it and return B (Index 1) which should have high similarity (~0.95)
    print("\nSearch for A (After Deletion) - Expecting match with B:")
    results = db.search(vec_a, top_k=1)
    
    print("Results:", results)
    
    if not results:
        print("FAIL: No match found! (The fix is not working)")
    elif results[0][0] == id_b:
        print(f"SUCCESS: Correctly matched Person B! Score: {results[0][1]:.4f}")
    else:
        print(f"FAIL: Matched unexpected person: {results[0][0]}")
    
    # 7. Test Negative Case (Threshold)
    # Create Person C (Random vector, likely dissimilar)
    vec_c = np.random.rand(512).astype(np.float32)
    vec_c /= np.linalg.norm(vec_c)
    
    # Ensure C is not similar to A (just in case)
    while np.dot(vec_a, vec_c) > 0.4:
         vec_c = np.random.rand(512).astype(np.float32)
         vec_c /= np.linalg.norm(vec_c)
         
    id_c = "person_c"
    db.add_embedding(id_c, vec_c)
    print(f"\nAdded Person C (Dissimilar, Score ~{np.dot(vec_a, vec_c):.4f}).")
    
    # Delete B so A has NO valid similar matches left
    print("Deleting Person B...")
    db.remove(id_b)
    
    print("Search for A again - Expecting NO MATCH (or match < 0.6):")
    # Note: VectorDB.search returns results regardless of threshold, 
    # but we want to simulate what the Service does.
    # The service logic is: if score < THRESHOLD, treat as no match.
    from app.config import SIMILARITY_THRESHOLD  # type: ignore
    
    results = db.search(vec_a, top_k=1)
    
    if not results:
        print("SUCCESS: No candidates returned.")
    else:
        pid, score = results[0]
        print(f"Candidate: {pid}, Score: {score:.4f}")
        if score < SIMILARITY_THRESHOLD:
            print(f"SUCCESS: Candidate found but Score {score:.4f} < {SIMILARITY_THRESHOLD} (Threshold enforced).")
        else:
            print(f"FAIL: False positive! matched {pid} with score {score:.4f}")

    # Cleanup
    db.reset()
    import os
    if os.path.exists('test_faiss.index'):
        try:
           os.remove('test_faiss.index')
           os.remove('test_faiss.index.mappings.npy')
        except: pass

if __name__ == "__main__":
    test_rematch_logic()
