import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from app.core.vector_db import get_vector_db

def test_embedding_update():
    print("Testing VectorDB update_embedding...")
    vector_db = get_vector_db()
    
    # Create fake embedding
    emb1 = np.random.rand(512).astype(np.float32)
    emb1 = emb1 / np.linalg.norm(emb1)
    
    # 1. Add
    pid = "test_person_v1"
    vector_db.add_embedding(pid, emb1)
    print(f"Added person {pid}")
    
    # 2. Retrieve
    retrieved = vector_db.get_embedding(pid)
    assert retrieved is not None, "Failed to retrieve embedding"
    assert np.allclose(emb1, retrieved, atol=1e-5), "Retrieved embedding mismatch"
    print("Retrieval check passed ✅")
    
    # 3. Update
    emb2 = np.random.rand(512).astype(np.float32)
    emb2 = emb2 / np.linalg.norm(emb2)
    
    print("Updating embedding...")
    vector_db.update_embedding(pid, emb2)
    
    # 4. Verify update
    updated = vector_db.get_embedding(pid)
    assert updated is not None
    
    # Should NOT match old
    dist_old = np.linalg.norm(updated - emb1)
    # Should match new
    dist_new = np.linalg.norm(updated - emb2)
    
    print(f"Distance to old: {dist_old}")
    print(f"Distance to new: {dist_new}")
    
    assert dist_old > 0.001, "Embedding did not change!"
    assert dist_new < 0.001, "Embedding did not update correctly!"
    
    print("Update check passed ✅")
    
    # Cleanup
    vector_db.remove(pid)
    print("Cleanup passed")

if __name__ == "__main__":
    test_embedding_update()
