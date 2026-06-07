"""
IVF-PQ Vector Index — Phase 1: Quantization Cascade

Three-stage quantized search:
  Stage 1 (Coarse Filter):  IVF partitions the vector space into `nlist` clusters.
                             At query time, only the `nprobe` nearest clusters are searched.
  Stage 2 (PQ Scan):        Within those clusters, vectors are stored compressed
                             (M bytes each instead of D×4 bytes). Approximate distances
                             are computed via precomputed lookup tables.
  Stage 3 (Exact Rerank):   The top `rerank_k` PQ candidates are re-scored against
                             their full float32 vectors to produce exact cosine scores.

Memory footprint per vector:
  Raw float32:  384 dims × 4 bytes = 1,536 bytes
  PQ-compressed: M bytes = 48 bytes  (32× smaller)
  Full vectors are kept separately for the rerank stage.

Important implementation detail:
  We L2-normalize all vectors BEFORE adding to FAISS, then use IndexFlatIP
  (inner product) as the coarse quantizer. This makes FAISS's inner product
  equivalent to cosine similarity, so the coarse filter, PQ scoring, and
  exact reranking all operate on the same metric.
"""

import logging
import pickle
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np

try:
    import faiss
except ImportError:
    faiss = None

from .base_index import BaseVectorIndex

log = logging.getLogger(__name__)


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize each row in-place-safe. Returns a new array."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return vectors / norms


class IVFPQIndex(BaseVectorIndex):
    """FAISS IVF-PQ index with three-stage search.

    Config parameters (from system_config.yaml → quantization):
        dimension (int): embedding dimension (384 for all-minilm)
        nlist (int):     number of IVF partitions. Rule: ~sqrt(N).
                         4096 works for up to ~16M vectors per shard.
        M (int):         PQ subvectors. Must divide dimension evenly.
                         384 / 8 = 48 subvectors → 48 bytes per vector.
        nbits (int):     bits per subquantizer. Always 8 (256 centroids).
        nprobe (int):    partitions to search per query. Default 64.
        rerank_k (int):  PQ candidates to exact-rerank. Default 500.
    """

    def __init__(
        self,
        dimension: int = 384,
        nlist: int = 4096,
        M: int = 48,
        nbits: int = 8,
        nprobe: int = 64,
        rerank_k: int = 500,
    ):
        if faiss is None:
            raise ImportError(
                "faiss-cpu is required for IVF-PQ indexing. "
                "Install with: pip install faiss-cpu"
            )

        self.dimension = dimension
        self.nlist = nlist
        self.M = M
        self.nbits = nbits
        self.nprobe = nprobe
        self.rerank_k = rerank_k

        # Internal state
        self._id_map: List[str] = []
        self._full_vectors: Optional[np.ndarray] = None  # normalized float32 for rerank
        self._index: Optional[faiss.IndexIVFPQ] = None
        self._is_trained = False

    # ── Index construction ──────────────────────────────────────────────

    def _create_faiss_index(self) -> "faiss.IndexIVFPQ":
        """Build the un-trained FAISS IVF-PQ index.

        Uses IndexFlatIP (inner product) as coarse quantizer.
        Since all vectors are L2-normalized before insertion,
        inner product = cosine similarity. This keeps the metric
        consistent across all three stages.
        """
        coarse_quantizer = faiss.IndexFlatIP(self.dimension)
        index = faiss.IndexIVFPQ(
            coarse_quantizer,
            self.dimension,
            self.nlist,
            self.M,
            self.nbits,
            faiss.METRIC_INNER_PRODUCT,  # cosine via normalized IP
        )
        index.nprobe = self.nprobe
        return index

    def build(self, vectors: np.ndarray, doc_ids: List[str]) -> None:
        """Train and populate the IVF-PQ index.

        All vectors are L2-normalized before training/adding so that
        inner product ≡ cosine similarity.

        FAISS requires at least 39 × nlist training samples.
        If the dataset is smaller, nlist is reduced automatically.
        """
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Expected shape (N, {self.dimension}), got {vectors.shape}"
            )

        N = vectors.shape[0]
        if N == 0:
            log.warning("Empty vector set — skipping IVF-PQ build")
            return

        # L2-normalize for cosine similarity
        vectors = _l2_normalize(vectors)

        # Adaptive nlist: FAISS needs >= 39 * nlist training vectors
        min_training = 39 * self.nlist
        if N < min_training:
            old_nlist = self.nlist
            self.nlist = max(1, N // 39)
            log.info("Adaptive nlist: %d → %d (dataset has %d vectors, need %d for %d cells)",
                     old_nlist, self.nlist, N, min_training, old_nlist)

        # M must divide dimension evenly. Validate.
        if self.dimension % self.M != 0:
            old_M = self.M
            # Find largest valid M <= old_M
            for m in range(old_M, 0, -1):
                if self.dimension % m == 0:
                    self.M = m
                    break
            log.info("Adjusted M: %d → %d (must divide dimension %d evenly)",
                     old_M, self.M, self.dimension)

        self._index = self._create_faiss_index()

        # Train on a subset for very large datasets
        train_size = min(N, 1_000_000)
        train_vectors = vectors[:train_size]

        log.info("Training IVF-PQ: %d vectors, dim=%d, nlist=%d, M=%d, nprobe=%d",
                 train_size, self.dimension, self.nlist, self.M, self.nprobe)
        self._index.train(train_vectors)
        self._is_trained = True

        # Add all vectors
        log.info("Adding %d vectors to IVF-PQ index...", N)
        self._index.add(vectors)

        self._id_map = list(doc_ids)
        self._full_vectors = vectors.copy()  # already normalized

        log.info("IVF-PQ index built: %d vectors indexed (faiss.ntotal=%d)",
                 N, self._index.ntotal)

    def add(self, vector: np.ndarray, doc_id: str) -> None:
        """Add a single vector to an already-trained index."""
        if not self._is_trained:
            raise RuntimeError("Index not trained — call build() first")

        v = np.ascontiguousarray(
            vector.reshape(1, -1), dtype=np.float32
        )
        v = _l2_normalize(v)

        self._index.add(v)
        self._id_map.append(doc_id)

        if self._full_vectors is not None:
            self._full_vectors = np.vstack([self._full_vectors, v])
        else:
            self._full_vectors = v

    # ── Search ──────────────────────────────────────────────────────────

    def search(self, query: np.ndarray, k: int, filter_bitmap=None) -> List[Tuple[str, float]]:
        """Three-stage search: IVF coarse → PQ approximate → exact rerank.

        Stage 1+2 are done by FAISS in a single call (returns approximate
        inner-product scores and candidate indices).
        Stage 3 re-computes exact dot products on the full normalized vectors.

        Returns (doc_id, cosine_similarity) tuples, descending.
        """
        if self._index is None or self._index.ntotal == 0:
            return []

        # Normalize query for cosine
        q = np.ascontiguousarray(query.reshape(1, -1), dtype=np.float32)
        q = _l2_normalize(q)

        # Stages 1+2: approximate search — ask for rerank_k candidates
        fetch_k = min(self.rerank_k, self._index.ntotal)
        _approx_scores, indices = self._index.search(q, fetch_k)

        # Filter padding (-1 = not found)
        valid = indices[0] != -1
        candidate_faiss_ids = indices[0][valid]

        if len(candidate_faiss_ids) == 0:
            return []

        # Stage 3: exact rerank using stored full vectors
        candidate_vectors = self._full_vectors[candidate_faiss_ids]  # (M, D), normalized
        exact_scores = candidate_vectors @ q[0]  # dot product = cosine (both normalized)

        # Top-k from the rerank candidates
        actual_k = min(k, len(exact_scores))
        top_idx = np.argpartition(-exact_scores, actual_k)[:actual_k]
        top_idx = top_idx[np.argsort(-exact_scores[top_idx])]

        results = []
        for i in top_idx:
            faiss_id = int(candidate_faiss_ids[i])
            if faiss_id < len(self._id_map):
                results.append((self._id_map[faiss_id], float(exact_scores[i])))

        return results

    # ── Persistence ─────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save to disk: FAISS index + id_map + full vectors."""
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._index, str(p / "faiss.index"))

        with open(p / "id_map.pkl", "wb") as f:
            pickle.dump(self._id_map, f, protocol=pickle.HIGHEST_PROTOCOL)

        np.save(str(p / "full_vectors.npy"), self._full_vectors)
        log.info("IVF-PQ index saved to %s (%d vectors)", path, self.size)

    def load(self, path: str) -> None:
        """Load from disk."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Index directory not found: {path}")

        self._index = faiss.read_index(str(p / "faiss.index"))
        self._index.nprobe = self.nprobe

        with open(p / "id_map.pkl", "rb") as f:
            self._id_map = pickle.load(f)

        self._full_vectors = np.load(str(p / "full_vectors.npy"))
        self._is_trained = True
        log.info("IVF-PQ index loaded from %s (%d vectors)", path, self.size)

    @property
    def size(self) -> int:
        return self._index.ntotal if self._index else 0
