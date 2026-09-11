
import os
import sys
import argparse
import numpy as np  # type: ignore
import cv2  # type: ignore

# Add parent directory to path to allow 'app' imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.vector_db import get_vector_db  # type: ignore
from app.core.encoder import get_encoder  # type: ignore
from app.config import SIMILARITY_THRESHOLD, FAISS_INDEX_PATH, SQLITE_PATH # type: ignore

def debug_matching():
    target_image_path = r"c:\projects\face_api\data\images\1b50434d-9aea-40f6-82de-83382d393363\b7bc89ce-f650-4d01-af2f-796d6c1af7b3.jpg"
    
    print(f"Debug Matching for: {target_image_path}")
    print(f"Current SIMILARITY_THRESHOLD: {SIMILARITY_THRESHOLD}")

    if not os.path.exists(target_image_path):
        print("Error: Image file not found.")
        return

    # Load resources
    print("Loading resources...")
    vector_db = get_vector_db()
    encoder = get_encoder()
    print(f"VectorDB has {vector_db.count} embeddings.")

    # Load and encode image
    print("Encoding image...")
    image = cv2.imread(target_image_path)
    if image is None:
        print("Error: Failed to load image with cv2.")
        return

    print(f"Image Shape: {image.shape}")

        
    embedding, det_info = encoder.detect_and_encode(image)
    
    if embedding is None:
        print("Error: No face detected or encoding failed.")
        return

    print("Image encoded successfully (Detection Mode).")
    
    # Search with detected embedding
    print("\n[Detection Mode] Searching VectorDB...")
    matches = vector_db.search(embedding, top_k=5)
    _print_matches(matches)

    # Test Cropped Mode (Bypass detection/alignment)
    print("\n[Cropped Mode] Encoding image as-is (force resize to 112x112)...")
    cropped_embedding = encoder.encode_cropped_face(image)
    
    if cropped_embedding is not None:
        print("\n[Cropped Mode] Searching VectorDB...")
        matches_cropped = vector_db.search(cropped_embedding, top_k=5)
        _print_matches(matches_cropped)
    else:
        print("[Cropped Mode] Encoding failed.")

def _print_matches(matches):
    if not matches:
        print("No matches found.")
    else:
        print("Top 5 matches:")
        for pid, score in matches:
            print(f" - Person ID: {pid}, Score: {score:.4f}")
            if score >= SIMILARITY_THRESHOLD:
                print(f"   -> WOULD MATCH (>= {SIMILARITY_THRESHOLD})")
            else:
                print(f"   -> WOULD NOT MATCH (< {SIMILARITY_THRESHOLD})")

if __name__ == "__main__":
    debug_matching()
