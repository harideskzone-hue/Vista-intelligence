"""
Vector Database Module
Handles FAISS index operations for fast similarity search.

Key design decisions:
- IndexFlatIP requires embeddings to be L2-normalised so inner product == cosine similarity.
- Deletion/update requires a full index rebuild (IndexFlat does not support remove).
  _rebuild_without() now does a genuine FAISS rebuild, not just dict cleanup.
- Detailed match logging helps diagnose threshold issues in production.
"""
import os
import logging
import numpy as np  # type: ignore
import faiss  # type: ignore

from app.config import FAISS_INDEX_PATH, EMBEDDING_DIM  # type: ignore

log = logging.getLogger("vector_db")


class VectorDB:
    """FAISS-based vector database for face embeddings."""

    def __init__(self):
        """Initialize or load the FAISS index."""
        self.dimension = EMBEDDING_DIM
        self.index_path = FAISS_INDEX_PATH

        # Bidirectional mapping between FAISS sequential IDs and person UUIDs
        self.id_to_person: dict[int, str] = {}
        self.person_to_id: dict[str, int] = {}
        self.next_id: int = 0

        if os.path.exists(self.index_path):
            self.load()
            log.info("VectorDB loaded: %d vectors, %d persons",
                     self.index.ntotal, len(self.person_to_id))
        else:
            self.index = faiss.IndexFlatIP(self.dimension)
            log.info("VectorDB initialised fresh (empty).")

    # ── Public API ──────────────────────────────────────────────────────────────

    def add_embedding(self, person_id: str, embedding: np.ndarray) -> int:
        """
        Add or update the embedding for a person.
        If the person already exists, the old vector is replaced via a full rebuild.

        Returns:
            New internal FAISS ID for the person.
        """
        embedding = np.array([embedding], dtype=np.float32)

        if person_id in self.person_to_id:
            # Person exists — rebuild index excluding their old entry first
            old_id = self.person_to_id[person_id]
            self._rebuild_without(old_id)

        internal_id = self.next_id
        self.index.add(embedding)
        self.id_to_person[internal_id] = person_id
        self.person_to_id[person_id] = internal_id
        self.next_id += 1

        return internal_id

    def get_embedding(self, person_id: str) -> np.ndarray | None:
        """Retrieve the stored embedding for a person."""
        if person_id not in self.person_to_id:
            return None

        internal_id = self.person_to_id[person_id]

        # Validate that the internal_id still exists in the index
        # (can be stale if bugs crept in from a previous run)
        if internal_id >= self.index.ntotal:
            log.warning("Stale internal_id %d for person %s (index.ntotal=%d) — skipping reconstruct",
                        internal_id, person_id, self.index.ntotal)
            return None

        try:
            return self.index.reconstruct(internal_id)
        except Exception as e:
            log.warning("reconstruct(%d) failed for person %s: %s", internal_id, person_id, e)
            return None

    def update_embedding(self, person_id: str, embedding: np.ndarray) -> bool:
        """Update the embedding for an existing person (triggers rebuild)."""
        if person_id not in self.person_to_id:
            return False
        self.add_embedding(person_id, embedding)
        return True

    def search(self, query_embedding: np.ndarray, top_k: int = 1) -> list[tuple[str, float]]:
        """
        Find the top-k most similar faces.

        Returns:
            List of (person_id, cosine_similarity) sorted descending.
        """
        if self.index.ntotal == 0:
            log.debug("search: index is empty, returning no results.")
            return []

        query = np.array([query_embedding], dtype=np.float32)
        # Cap search_k to avoid unnecessary overhead on large indices
        search_k = min(max(top_k * 5, 10), self.index.ntotal)
        scores, indices = self.index.search(query, search_k)

        results: list[tuple[str, float]] = []
        for score, idx in zip(scores[0], indices[0]):  # type: ignore
            if idx >= 0 and idx in self.id_to_person:
                pid = self.id_to_person[idx]
                log.debug("  candidate: person=%s  similarity=%.4f  faiss_idx=%d",
                          pid, float(score), idx)
                results.append((pid, float(score)))
                if len(results) >= top_k:
                    break

        if results:
            best_pid, best_sim = results[0]
            log.info("search result: best_person=%s  similarity=%.4f", best_pid, best_sim)
        else:
            log.info("search result: no valid candidates found")

        return results

    def remove(self, person_id: str) -> bool:
        """Remove a person from the index (triggers rebuild)."""
        if person_id not in self.person_to_id:
            return False
        old_id = self.person_to_id[person_id]
        self._rebuild_without(old_id)
        return True

    # ── Persistence ─────────────────────────────────────────────────────────────

    def save(self):
        """Atomically save the index and mappings to disk."""
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(self.index, self.index_path)

        mapping_path = self.index_path + '.mappings.npy'
        np.save(mapping_path, {
            'id_to_person': self.id_to_person,
            'person_to_id': self.person_to_id,
            'next_id': self.next_id
        }, allow_pickle=True)
        log.debug("VectorDB saved: %d vectors on disk.", self.index.ntotal)

    def load(self):
        """Load the index and mappings from disk."""
        self.index = faiss.read_index(self.index_path)

        mapping_path = self.index_path + '.mappings.npy'
        if os.path.exists(mapping_path):
            mappings = np.load(mapping_path, allow_pickle=True).item()
            self.id_to_person = mappings.get('id_to_person', {})
            self.person_to_id = mappings.get('person_to_id', {})
            self.next_id = mappings.get('next_id', 0)

            # Integrity check: warn if dicts disagree with actual FAISS ntotal
            actual = self.index.ntotal
            mapped = len(self.id_to_person)
            if actual != mapped:
                log.warning(
                    "Index integrity mismatch: FAISS ntotal=%d but id_to_person has %d entries. "
                    "Rebuilding from mappings...", actual, mapped)
                self._compact_rebuild()
        else:
            log.warning("No mappings file found alongside FAISS index — starting with empty maps.")
            self.id_to_person = {}
            self.person_to_id = {}
            self.next_id = 0

    # ── Management ──────────────────────────────────────────────────────────────

    def reset(self):
        """Clear the index and all mappings (wipes disk files too)."""
        self.id_to_person = {}
        self.person_to_id = {}
        self.next_id = 0
        self.index = faiss.IndexFlatIP(self.dimension)

        for path in [self.index_path, self.index_path + '.mappings.npy']:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass
        log.info("VectorDB reset complete.")

    @property
    def count(self) -> int:
        """Number of embeddings in the index."""
        return self.index.ntotal

    # ── Private helpers ─────────────────────────────────────────────────────────

    def _rebuild_without(self, exclude_id: int):
        """
        Perform a genuine FAISS index rebuild, omitting exclude_id.

        This is the correct way to 'delete' a vector from IndexFlatIP which
        does not support removal natively.  After the rebuild, all internal IDs
        are reassigned sequentially; person_to_id and id_to_person are updated.
        """
        retained_pids: list[str] = [
            pid for iid, pid in sorted(self.id_to_person.items())
            if iid != exclude_id
        ]

        # Reconstruct kept vectors in original order
        kept_vectors: list[np.ndarray] = []
        for iid, pid in sorted(self.id_to_person.items()):
            if iid == exclude_id:
                continue
            try:
                if iid < self.index.ntotal:
                    vec = self.index.reconstruct(iid)
                    kept_vectors.append(vec)
                else:
                    log.warning("_rebuild_without: iid %d out of bounds (ntotal=%d), zeroing.",
                                iid, self.index.ntotal)
                    kept_vectors.append(np.zeros(self.dimension, dtype=np.float32))
            except Exception as e:
                log.warning("_rebuild_without: reconstruct(%d) error: %s — zeroing.", iid, e)
                kept_vectors.append(np.zeros(self.dimension, dtype=np.float32))

        # Build fresh index
        new_index = faiss.IndexFlatIP(self.dimension)
        new_id_to_person: dict[int, str] = {}
        new_person_to_id: dict[str, int] = {}

        if kept_vectors:
            matrix = np.stack(kept_vectors, axis=0).astype(np.float32)
            new_index.add(matrix)
            for new_iid, pid in enumerate(retained_pids):
                new_id_to_person[new_iid] = pid
                new_person_to_id[pid] = new_iid

        self.index = new_index
        self.id_to_person = new_id_to_person
        self.person_to_id = new_person_to_id
        self.next_id = len(retained_pids)
        log.debug("_rebuild_without(%d): index rebuilt with %d entries.", exclude_id, self.next_id)

    def _compact_rebuild(self):
        """
        Re-sync index from disk when ntotal disagrees with the mapping dict.
        Keeps only entries present in both the FAISS file and id_to_person.
        """
        valid_ids = [iid for iid in sorted(self.id_to_person) if iid < self.index.ntotal]
        new_index = faiss.IndexFlatIP(self.dimension)
        new_id_to_person: dict[int, str] = {}
        new_person_to_id: dict[str, int] = {}

        for new_iid, old_iid in enumerate(valid_ids):
            pid = self.id_to_person[old_iid]
            try:
                vec = np.array([self.index.reconstruct(old_iid)], dtype=np.float32)
                new_index.add(vec)
                new_id_to_person[new_iid] = pid
                new_person_to_id[pid] = new_iid
            except Exception as e:
                log.warning("_compact_rebuild: skipping iid %d (%s): %s", old_iid, pid, e)

        self.index = new_index
        self.id_to_person = new_id_to_person
        self.person_to_id = new_person_to_id
        self.next_id = len(new_id_to_person)
        log.info("_compact_rebuild: compacted to %d valid entries.", self.next_id)


# ── Global singleton ────────────────────────────────────────────────────────────
_vector_db: VectorDB | None = None


def get_vector_db() -> VectorDB:
    """Get or create the global VectorDB instance."""
    global _vector_db
    if _vector_db is None:
        _vector_db = VectorDB()
    return _vector_db
