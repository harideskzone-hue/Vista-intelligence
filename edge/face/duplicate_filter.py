"""
Duplicate Filter: Identity-aware duplicate image rejection.
Prevents filling storage with identical or near-identical frames.
Uses SHA-256 for exact matches and pHash for perceptual near-duplicates.
"""
from __future__ import annotations

import hashlib
import cv2
import numpy as np

def compute_sha256(image_bytes: bytes) -> str:
    """Computes SHA-256 hash for exact duplicate detection."""
    return hashlib.sha256(image_bytes).hexdigest()

def compute_phash(image: np.ndarray) -> str:
    """Computes a perceptual hash (dHash) for near-duplicate detection."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
        
    # Resize to 9x8 to compute 8x8 difference hash
    resized = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    
    # Compute differences between adjacent pixels
    diff = resized[:, 1:] > resized[:, :-1]
    
    # Convert 64 boolean values to a 64-bit integer, then to hex string
    hash_value = 0
    for i, v in enumerate(diff.flatten()):
        if v:
            hash_value |= (1 << i)
            
    return f"{hash_value:016x}"

def phash_distance(hash1: str, hash2: str) -> int:
    """Computes the Hamming distance between two hex pHashes."""
    h1 = int(hash1, 16)
    h2 = int(hash2, 16)
    # Count set bits in XOR
    return bin(h1 ^ h2).count('1')


class DuplicateFilter:
    """
    Filters exact and near-duplicate images.
    Near-duplicate checking is strictly identity-aware (per person_id) to avoid
    accidentally discarding valid frames of visually similar distinct people.
    """
    
    def __init__(self, phash_threshold: int = 5):
        self.phash_threshold = phash_threshold
        # Tracks recent perceptual hashes per person: person_id -> list[str]
        self.recent_phashes: dict[str, list[str]] = {}

    def is_duplicate(self, person_id: str, image_bytes: bytes, image_np: np.ndarray, exact_exists_fn) -> tuple[bool, str]:
        """
        Check if the image is a duplicate.
        exact_exists_fn is a callable that checks the database for the SHA-256 hash.
        
        Returns:
            (is_duplicate, reason)
        """
        # 1. Exact Duplicate (Global)
        sha256_hash = compute_sha256(image_bytes)
        if exact_exists_fn(sha256_hash):
            return True, "EXACT_DUPLICATE"
            
        # 2. Near Duplicate (Per-Identity)
        phash = compute_phash(image_np)
        if person_id in self.recent_phashes:
            for recent_hash in self.recent_phashes[person_id]:
                if phash_distance(phash, recent_hash) <= self.phash_threshold:
                    return True, "NEAR_DUPLICATE"
                    
        return False, "UNIQUE"

    def register_image(self, person_id: str, image_np: np.ndarray) -> None:
        """
        Register a successfully saved image's pHash for future filtering.
        Keeps a rolling window of recent hashes per person.
        """
        phash = compute_phash(image_np)
        if person_id not in self.recent_phashes:
            self.recent_phashes[person_id] = []
        
        self.recent_phashes[person_id].append(phash)
        
        # Keep history bounded to avoid memory leaks
        if len(self.recent_phashes[person_id]) > 50:
            self.recent_phashes[person_id].pop(0)
