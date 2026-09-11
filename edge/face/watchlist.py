"""
Watchlist: FAISS-accelerated retrieval index for face embeddings.
FAISS is used for retrieval only. The SQLite database is the strict source of truth.
If FAISS and SQLite are inconsistent, the system fails closed.
"""
from __future__ import annotations

import os
import json
import logging
import numpy as np
import faiss

from edge.face.identity_manager import IdentityManager

log = logging.getLogger("watchlist")

class WatchlistIntegrityError(Exception):
    """Raised when FAISS index state does not match the SQLite database."""
    pass

class Watchlist:
    def __init__(self, identity_manager: IdentityManager, index_dir: str, dimension: int = 512):
        self.identity_manager = identity_manager
        self.index_dir = index_dir
        self.dimension = dimension
        
        self.index_path = os.path.join(index_dir, "faiss.index")
        self.mapping_path = os.path.join(index_dir, "mapping.json")
        
        self.index: faiss.IndexFlatIP | None = None
        # Maps FAISS internal integer ID to SQLite embedding_id (str)
        self.faiss_to_emb: dict[int, str] = {}
        
        self.load_or_rebuild()

    def load_or_rebuild(self) -> None:
        """
        Loads the index from disk if present, then validates against the DB.
        If missing or corrupted/inconsistent, rebuilds from the DB.
        """
        os.makedirs(self.index_dir, exist_ok=True)
        db_embeddings = self.identity_manager.get_all_embeddings()
        
        if os.path.exists(self.index_path) and os.path.exists(self.mapping_path):
            try:
                self.index = faiss.read_index(self.index_path)
                with open(self.mapping_path, 'r') as f:
                    str_mapping = json.load(f)
                    self.faiss_to_emb = {int(k): v for k, v in str_mapping.items()}
                
                self.validate_integrity(db_embeddings)
                log.info(f"Watchlist loaded successfully: {self.index.ntotal} embeddings.")
                return
            except Exception as e:
                log.warning(f"Watchlist integrity or load failure: {e}. Rebuilding index...")
        
        self.rebuild_index(db_embeddings)

    def validate_integrity(self, db_embeddings: list) -> None:
        """
        Strictly validate that FAISS exactly matches the DB.
        Raises WatchlistIntegrityError if inconsistent.
        """
        if self.index is None:
            raise WatchlistIntegrityError("FAISS index is not initialized.")
            
        if self.index.ntotal != len(self.faiss_to_emb):
            raise WatchlistIntegrityError(f"FAISS count ({self.index.ntotal}) != mapping count ({len(self.faiss_to_emb)})")
            
        active_db_embeddings = []
        for emb in db_embeddings:
            person = self.identity_manager.get_person(emb.person_id)
            if not person:
                raise WatchlistIntegrityError(f"DB Embedding {emb.embedding_id} points to missing Person {emb.person_id}.")
            if person.status == 'ACTIVE':
                active_db_embeddings.append(emb)
                
        if self.index.ntotal != len(active_db_embeddings):
            raise WatchlistIntegrityError(f"FAISS count ({self.index.ntotal}) != Active DB count ({len(active_db_embeddings)})")
            
        db_emb_ids = {e.embedding_id for e in active_db_embeddings}
        faiss_emb_ids = set(self.faiss_to_emb.values())
        
        if db_emb_ids != faiss_emb_ids:
            raise WatchlistIntegrityError("FAISS mappings do not strictly match DB embeddings.")
            


    def rebuild_index(self, db_embeddings: list | None = None) -> None:
        """
        Rebuild the FAISS index entirely from the DB source of truth.
        """
        if db_embeddings is None:
            db_embeddings = self.identity_manager.get_all_embeddings()
            
        self.index = faiss.IndexFlatIP(self.dimension)
        self.faiss_to_emb = {}
        
        if db_embeddings:
            vectors = []
            idx_counter = 0
            for emb in db_embeddings:
                person = self.identity_manager.get_person(emb.person_id)
                if not person:
                    raise WatchlistIntegrityError(f"Cannot build index: Person {emb.person_id} missing from DB.")
                
                # Exclude DISABLED persons from FAISS index completely
                if person.status != 'ACTIVE':
                    continue
                
                vec = np.frombuffer(emb.embedding_vector, dtype=np.float32)
                if vec.shape[0] != self.dimension:
                    raise ValueError(f"Embedding {emb.embedding_id} has wrong dimension: {vec.shape[0]}")
                vectors.append(vec)
                self.faiss_to_emb[idx_counter] = emb.embedding_id
                idx_counter += 1
                
            if vectors:
                matrix = np.stack(vectors, axis=0)
                self.index.add(matrix)
            else:
                log.info("No active embeddings found. Index is empty.")
            
        self.save()
        log.info(f"Watchlist rebuilt with {len(db_embeddings)} embeddings.")

    def save(self) -> None:
        """Save the index and mappings to disk."""
        if self.index is not None:
            faiss.write_index(self.index, self.index_path)
            with open(self.mapping_path, 'w') as f:
                json.dump(self.faiss_to_emb, f)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[tuple[str, str, float]]:
        """
        Retrieves the top-K candidates from FAISS.
        
        Returns:
            List of (person_id, embedding_id, similarity)
            
        Note:
            This method ONLY retrieves candidates. It does NOT make an identity match decision.
        """
        if self.index is None or self.index.ntotal == 0:
            return []
            
        query = np.array([query_embedding], dtype=np.float32)
        search_k = min(top_k, self.index.ntotal)
        
        scores, indices = self.index.search(query, search_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx in self.faiss_to_emb:
                emb_id = self.faiss_to_emb[idx]
                
                # Retrieve from SQLite to ensure source-of-truth mapping
                emb = self.identity_manager.get_embedding(emb_id)
                if not emb:
                    # Fail closed on inconsistency
                    raise WatchlistIntegrityError(f"FAISS mapped to embedding {emb_id} but it is missing from DB.")
                    
                person = self.identity_manager.get_person(emb.person_id)
                if not person:
                    # Fail closed on inconsistency
                    raise WatchlistIntegrityError(f"Embedding {emb_id} points to missing person {emb.person_id}.")
                    
                if person.status == 'ACTIVE':
                    results.append((person.person_id, emb_id, float(score)))
                
        return results
