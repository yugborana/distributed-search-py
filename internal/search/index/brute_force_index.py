"""
Brute-Force Vector Index — fallback when quantization is disabled.

Does a full numpy dot-product scan over all vectors. Fast for small datasets
(<100K vectors) but doesn't scale. Used as the default when FAISS isn't needed.

Vectors are pre-normalized at build time so search is a simple matrix multiply.
"""

import logging
import pickle
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np

from .base_index import BaseVectorIndex

log = logging.getLogger(__name__)


def _normalize(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize each row. Returns a new array."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)  # avoid division by zero
    return vectors / norms


class BruteForceIndex(BaseVectorIndex):
    """Brute-force cosine similarity index backed by numpy.

    Functionally equivalent to what hybrid.py's cosine_similarity() was doing
    per-hit, but batched for speed. Pre-normalizes all vectors at build time
    so search is a single matrix multiply.
    """

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self._vectors: Optional[np.ndarray] = None  # (N, D), pre-normalized
        self._id_map: List[str] = []

    def build(self, vectors: np.ndarray, doc_ids: List[str]) -> None:
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        assert vectors.ndim == 2 and vectors.shape[1] == self.dimension
        assert len(doc_ids) == vectors.shape[0]

        self._vectors = _normalize(vectors)
        self._id_map = list(doc_ids)
        log.info("BruteForce index built: %d vectors (dim=%d)", len(doc_ids), self.dimension)

    def add(self, vector: np.ndarray, doc_id: str) -> None:
        v = _normalize(vector.reshape(1, -1).astype(np.float32))
        if self._vectors is not None:
            self._vectors = np.vstack([self._vectors, v])
        else:
            self._vectors = v
        self._id_map.append(doc_id)

    def search(self, query: np.ndarray, k: int, filter_bitmap=None) -> List[Tuple[str, float]]:
        if self._vectors is None or len(self._id_map) == 0:
            return []

        q = query.reshape(-1).astype(np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return []
        q = q / q_norm

        scores = self._vectors @ q  # cosine similarities (vectors are pre-normalized)

        top_k = min(k, len(scores))
        if top_k <= 0:
            return []

        # argpartition is O(N) vs O(N log N) for full sort — much faster for large N
        top_indices = np.argpartition(-scores, top_k)[:top_k]
        top_indices = top_indices[np.argsort(-scores[top_indices])]

        return [(self._id_map[i], float(scores[i])) for i in top_indices]

    def save(self, path: str) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        np.save(str(p / "vectors.npy"), self._vectors)
        with open(p / "id_map.pkl", "wb") as f:
            pickle.dump(self._id_map, f, protocol=pickle.HIGHEST_PROTOCOL)
        log.info("BruteForce index saved to %s (%d vectors)", path, self.size)

    def load(self, path: str) -> None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Index path does not exist: {path}")
        self._vectors = np.load(str(p / "vectors.npy"))
        with open(p / "id_map.pkl", "rb") as f:
            self._id_map = pickle.load(f)
        log.info("BruteForce index loaded from %s (%d vectors)", path, self.size)

    @property
    def size(self) -> int:
        return len(self._id_map)
