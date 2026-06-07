"""
DiskANN (Vamana Graph) Vector Index — Phase 2

Replaces the in-RAM full_vectors reranking from Phase 1 (IVF-PQ) with an
SSD-backed navigable graph. At scale (125M vectors/shard), holding full
float32 vectors in RAM requires 192GB — impossible. DiskANN keeps the graph
on SSD and only caches frequently-visited "hotspot" nodes in RAM.

How it works:
  1. Build: constructs a Vamana graph where each node stores its full vector
     + a list of ~R neighbor IDs. All on disk.
  2. Search: starts at the graph's medoid, greedily walks edges toward the
     query vector. Each hop reads one 4KB disk sector (vector + neighbors).
     With NVMe SSDs doing 100K+ IOPS, this is fast enough.
  3. Hotspot cache: the most-visited nodes (top few hops from every entry
     point) are cached in RAM. After 2–3 hops in RAM, search goes to disk
     only for "cold" nodes.

Architecture after Phase 2:
  IVF-PQ coarse filter → PQ candidates → DiskANN on-disk rerank → top-K
  RAM: IVF centroids + PQ codes + DiskANN hotspot cache (~4–8 GB)
  SSD: DiskANN graph file (nodes + neighbors + full float32 vectors)

Online writes:
  DiskANN does not support true online insertion. New vectors go into a
  small in-memory FAISS flat "fresh buffer". Queries search both the main
  DiskANN index and the fresh buffer, merging results. When the buffer
  exceeds a threshold (100K docs), a background merge rebuilds the full
  DiskANN index.

Adapted for our codebase:
  - Uses "mips" metric with L2-normalized vectors (cosine similarity)
  - Vectors come from Tantivy's stored title_vector bytes
  - Sits alongside BM25 — coordinator fuses both via RRF/weighted
  - Graceful fallback if diskannpy is not installed
"""

import logging
import pickle
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np

try:
    import diskannpy
except ImportError:
    diskannpy = None

try:
    import faiss
except ImportError:
    faiss = None

from .base_index import BaseVectorIndex

log = logging.getLogger(__name__)


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize each row. Returns a new array."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return vectors / norms


class DiskANNIndex(BaseVectorIndex):
    """DiskANN Vamana graph index — SSD-backed vector search.

    Config parameters (from system_config.yaml → index):
        dimension (int):       embedding dimension (384)
        data_dir (str):        directory for DiskANN files on disk
        R (int):               graph degree (max neighbors per node). Default 64.
                               32 → ~90% recall, 64 → ~95%, 96 → ~99%.
        L_build (int):         search list size during build. Default 100.
                               Higher = better graph quality, slower build.
        B (float):             SSD read budget per query in GB. Default 0.003 (3MB).
        M_pq (int):            PQ bytes for in-memory compressed vectors during
                               beam search scoring. Default 64.
        cache_ram_gb (float):  RAM for hotspot cache. Default 4.0 GB.
        beam_width (int):      beam search width at query time. Default 8.
        search_L (int):        search-time complexity (L parameter). Default 100.
    """

    # Threshold for merging fresh buffer into full DiskANN index
    FRESH_BUFFER_MERGE_THRESHOLD = 100_000

    def __init__(
        self,
        dimension: int = 384,
        data_dir: str = "/data/diskann",
        R: int = 64,
        L_build: int = 100,
        B: float = 0.003,
        M_pq: int = 64,
        cache_ram_gb: float = 4.0,
        beam_width: int = 8,
        search_L: int = 100,
    ):
        if diskannpy is None:
            raise ImportError(
                "diskannpy is required for DiskANN indexing. "
                "Install with: pip install diskannpy"
            )

        self.dimension = dimension
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.R = R
        self.L_build = L_build
        self.B = B
        self.M_pq = M_pq
        self.cache_ram_gb = cache_ram_gb
        self.beam_width = beam_width
        self.search_L = search_L

        # Internal state
        self._id_map: List[str] = []
        self._index_handle = None
        self._n_vectors: int = 0

        # Fresh buffer for online adds (DiskANN can't do true online insert)
        self._fresh_vectors: List[np.ndarray] = []
        self._fresh_ids: List[str] = []
        self._fresh_faiss: Optional[object] = None  # faiss.IndexFlatIP for fresh buffer search

    # ── Build ───────────────────────────────────────────────────────────

    def build(self, vectors: np.ndarray, doc_ids: List[str]) -> None:
        """Build the DiskANN Vamana graph from scratch.

        This is slow for large datasets (minutes to hours) but only runs once.
        All vectors are L2-normalized so we can use MIPS metric ≡ cosine.
        """
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Expected shape (N, {self.dimension}), got {vectors.shape}"
            )

        N = vectors.shape[0]
        if N == 0:
            log.warning("Empty vector set — skipping DiskANN build")
            return

        # L2-normalize for cosine via MIPS
        vectors = _l2_normalize(vectors)

        self._id_map = list(doc_ids)
        self._n_vectors = N

        # Write vectors in DiskANN's fbin format: [uint32 N][uint32 D][float32 × N × D]
        raw_vector_file = str(self.data_dir / "vectors.fbin")
        self._write_fbin(vectors, raw_vector_file)

        log.info("Building DiskANN index: %d vectors, R=%d, L_build=%d, M_pq=%d",
                 N, self.R, self.L_build, self.M_pq)

        diskannpy.build_disk_index(
            data=raw_vector_file,
            metric="mips",                     # cosine via normalized MIPS
            vector_dtype=np.float32,
            index_directory=str(self.data_dir),
            index_prefix="diskann_index",
            complexity=self.L_build,
            graph_degree=self.R,
            search_memory_maximum=self.cache_ram_gb,
            build_memory_maximum=self.cache_ram_gb * 2,
            num_threads=0,                     # 0 = all CPU cores
            pq_disk_bytes=self.M_pq,
        )

        # Save id_map
        with open(self.data_dir / "id_map.pkl", "wb") as f:
            pickle.dump(self._id_map, f, protocol=pickle.HIGHEST_PROTOCOL)

        log.info("DiskANN index built: %d vectors", N)

        # Clear any stale search handle — force reload on next search
        self._index_handle = None

    def _load_search_index(self):
        """Load the built DiskANN index for searching."""
        # Estimate how many nodes fit in cache
        # Each node ≈ dimension*4 bytes (vector) + R*4 bytes (neighbors)
        bytes_per_node = self.dimension * 4 + self.R * 4
        max_cache_nodes = int(self.cache_ram_gb * 1e9 / bytes_per_node)

        self._index_handle = diskannpy.DiskIndex(
            metric="mips",
            vector_dtype=np.float32,
            index_directory=str(self.data_dir),
            index_prefix="diskann_index",
            num_threads=4,
            num_nodes_to_cache=max_cache_nodes,
            cache_mechanism=1,  # 1 = frequency-based (hotspot nodes)
        )
        log.info("DiskANN search index loaded (cache=%d nodes)", max_cache_nodes)

    # ── Search ──────────────────────────────────────────────────────────

    def search(self, query: np.ndarray, k: int, filter_bitmap=None) -> List[Tuple[str, float]]:
        """Beam search through the DiskANN graph + fresh buffer.

        Searches both the main on-disk DiskANN index and the in-memory
        fresh buffer, then merges and returns the top-k results.
        """
        if self._n_vectors == 0 and len(self._fresh_ids) == 0:
            return []

        q = np.ascontiguousarray(query.reshape(1, -1), dtype=np.float32)
        q = _l2_normalize(q)

        results = []

        # Search main DiskANN index
        if self._n_vectors > 0:
            if self._index_handle is None:
                self._load_search_index()

            distances, indices = self._index_handle.search(
                queries=q,
                k_neighbors=k,
                complexity=self.search_L,
                beam_width=self.beam_width,
            )

            for faiss_id, score in zip(indices[0], distances[0]):
                if faiss_id == -1:
                    continue
                if faiss_id < len(self._id_map):
                    # MIPS returns inner product directly (higher = more similar)
                    results.append((self._id_map[faiss_id], float(score)))

        # Search fresh buffer (small FAISS flat index)
        if self._fresh_faiss is not None and self._fresh_faiss.ntotal > 0:
            fresh_k = min(k, self._fresh_faiss.ntotal)
            f_scores, f_indices = self._fresh_faiss.search(q, fresh_k)
            for idx, score in zip(f_indices[0], f_scores[0]):
                if idx == -1:
                    continue
                fresh_doc_id = self._fresh_ids[idx]
                results.append((fresh_doc_id, float(score)))

        # Merge: sort by score descending, return top-k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

    # ── Online adds ─────────────────────────────────────────────────────

    def add(self, vector: np.ndarray, doc_id: str) -> None:
        """Buffer a new vector for later merge into DiskANN.

        New vectors go into a small FAISS flat index so they're immediately
        searchable. When the buffer exceeds FRESH_BUFFER_MERGE_THRESHOLD,
        a full rebuild is triggered.
        """
        v = np.ascontiguousarray(vector.reshape(1, -1), dtype=np.float32)
        v = _l2_normalize(v)

        self._fresh_vectors.append(v[0])
        self._fresh_ids.append(doc_id)
        self._id_map.append(doc_id)

        # Add to FAISS flat index for immediate searchability
        if faiss is not None:
            if self._fresh_faiss is None:
                self._fresh_faiss = faiss.IndexFlatIP(self.dimension)
            self._fresh_faiss.add(v)

        # Trigger merge when buffer is large enough
        if len(self._fresh_vectors) >= self.FRESH_BUFFER_MERGE_THRESHOLD:
            self._merge_fresh_buffer()

    def _merge_fresh_buffer(self):
        """Merge fresh buffer into the main DiskANN index by rebuilding.

        In production this would be a background job. For now, a synchronous
        rebuild from the combined vectors (existing + fresh).
        """
        log.info("Merging %d fresh vectors into DiskANN index...",
                 len(self._fresh_vectors))

        # Load existing vectors from the fbin file
        fbin_path = self.data_dir / "vectors.fbin"
        if fbin_path.exists():
            existing = self._read_fbin(str(fbin_path))
        else:
            existing = np.empty((0, self.dimension), dtype=np.float32)

        # Combine
        new_vecs = np.array(self._fresh_vectors, dtype=np.float32)
        combined = np.vstack([existing, new_vecs]) if existing.shape[0] > 0 else new_vecs
        combined_ids = list(self._id_map)  # already includes fresh IDs from add()

        # Rebuild
        self._fresh_vectors = []
        self._fresh_ids = []
        self._fresh_faiss = None
        self.build(combined, combined_ids)

    # ── Persistence ─────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """DiskANN files are already on disk. Save id_map and fresh buffer."""
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)

        with open(p / "diskann_id_map.pkl", "wb") as f:
            pickle.dump(self._id_map, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Save fresh buffer state if any
        if self._fresh_vectors:
            np.save(str(p / "fresh_vectors.npy"),
                    np.array(self._fresh_vectors, dtype=np.float32))
            with open(p / "fresh_ids.pkl", "wb") as f:
                pickle.dump(self._fresh_ids, f)

        log.info("DiskANN index state saved to %s", path)

    def load(self, path: str) -> None:
        """Load id_map and restore DiskANN search handle."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"DiskANN path not found: {path}")

        with open(p / "diskann_id_map.pkl", "rb") as f:
            self._id_map = pickle.load(f)

        self._n_vectors = len(self._id_map)

        # Restore fresh buffer if it was saved
        fresh_vec_path = p / "fresh_vectors.npy"
        if fresh_vec_path.exists():
            fresh_vecs = np.load(str(fresh_vec_path))
            self._fresh_vectors = [v for v in fresh_vecs]
            with open(p / "fresh_ids.pkl", "rb") as f:
                self._fresh_ids = pickle.load(f)
            # Rebuild fresh FAISS index
            if faiss is not None and len(self._fresh_vectors) > 0:
                self._fresh_faiss = faiss.IndexFlatIP(self.dimension)
                self._fresh_faiss.add(fresh_vecs)

        # Lazy-load the DiskANN search handle on first search()
        self._index_handle = None
        log.info("DiskANN state loaded from %s (%d vectors, %d fresh)",
                 path, self._n_vectors, len(self._fresh_ids))

    # ── Binary I/O helpers ──────────────────────────────────────────────

    @staticmethod
    def _write_fbin(vectors: np.ndarray, path: str):
        """Write vectors in DiskANN's fbin format: [uint32 N][uint32 D][float32 data]."""
        N, D = vectors.shape
        with open(path, "wb") as f:
            np.array([N, D], dtype=np.uint32).tofile(f)
            vectors.tofile(f)

    @staticmethod
    def _read_fbin(path: str) -> np.ndarray:
        """Read vectors from DiskANN's fbin format."""
        with open(path, "rb") as f:
            header = np.fromfile(f, dtype=np.uint32, count=2)
            N, D = int(header[0]), int(header[1])
            vectors = np.fromfile(f, dtype=np.float32, count=N * D)
            return vectors.reshape(N, D)

    @property
    def size(self) -> int:
        return self._n_vectors + len(self._fresh_ids)
