"""
I/O Deduplication Reader — Phase 4

Deduplicates concurrent disk reads for the same vector, and maintains an
LRU cache of recently-read vectors to avoid repeat I/O entirely.

Problem:
  At high QPS, many concurrent queries want the same popular vectors.
  Without dedup, 500 concurrent queries needing vector #42 each trigger
  their own SSD read — 500 reads of the same 1.5KB data.
  With dedup: 1 read, broadcast to all 500 waiters.

FusionANNS (USENIX FAST '25 best paper) showed "redundant-aware I/O
deduplication" yields 30–50% fewer SSD reads under load.

Two modes of operation:

  1. FILE-BACKED (for DiskANN / flat binary vector files):
     Reads from a vectors.fbin file in DiskANN format:
     [uint32 N][uint32 D][float32 × N × D]
     Each vector is at offset: 8 + vector_id × (dim × 4)

  2. DOC-ID-KEYED (for Tantivy scored_search):
     A general-purpose LRU cache keyed by string doc_id.
     The caller provides the vector; the reader caches it.
     Under concurrent queries, the same popular doc vectors
     are returned from cache instead of being re-parsed from
     Tantivy's stored fields.

Adapted for our codebase:
  - Our shard doesn't have a separate shard_worker.py — the searcher.py
    shard process uses this directly.
  - Provides BOTH async (for concurrent coroutines) and sync (for numpy
    batch operations) access patterns.
  - The LRU eviction uses OrderedDict which is O(1) for all operations.
  - asyncio.Future-based dedup means zero extra disk reads for in-flight
    requests — all waiters share the same result.
"""

import asyncio
import logging
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Optional, List

import numpy as np

log = logging.getLogger(__name__)


class IODedupReader:
    """Deduplicates concurrent disk reads and caches recently-read vectors.

    When 500 queries all need vector ID #42, only 1 disk read happens.
    The other 499 await the same asyncio.Future.

    Args:
        vector_file_path: Path to flat binary vector file (fbin format).
                          None if using doc-id-keyed mode only.
        vector_dim:       Dimension of vectors (384 for all-minilm).
        cache_size_mb:    RAM budget for LRU cache. 512 MB ≈ 349K vectors.
    """

    def __init__(
        self,
        vector_file_path: Optional[str] = None,
        vector_dim: int = 384,
        cache_size_mb: int = 512,
    ):
        self.vector_file_path = vector_file_path
        self.vector_dim = vector_dim
        self.vector_bytes = vector_dim * 4  # float32

        # In-flight reads: key → asyncio.Future
        # If a read for a key is already in progress, new requesters
        # await the same future instead of starting a new read.
        self._in_flight: dict[str, asyncio.Future] = {}

        # LRU cache: key → np.ndarray (shape (dim,), dtype float32)
        cache_capacity = max(1, (cache_size_mb * 1024 * 1024) // self.vector_bytes)
        self._lru: OrderedDict[str, np.ndarray] = OrderedDict()
        self._lru_capacity: int = cache_capacity

        # Thread lock for sync access (IVF-PQ reranking runs in threads)
        self._sync_lock = threading.Lock()

        # File handle for binary vector file (lazy-opened)
        self._file_handle = None

        # Stats for monitoring
        self.stats_hits = 0
        self.stats_misses = 0
        self.stats_deduped = 0

    # ── Sync API (for IVF-PQ reranking / batch numpy operations) ────────

    def get_cached(self, key: str) -> Optional[np.ndarray]:
        """Get a vector from cache, or None if not cached. Thread-safe."""
        with self._sync_lock:
            if key in self._lru:
                self._lru.move_to_end(key)
                self.stats_hits += 1
                return self._lru[key]
            self.stats_misses += 1
            return None

    def put(self, key: str, vector: np.ndarray) -> None:
        """Store a vector in the LRU cache. Thread-safe."""
        with self._sync_lock:
            self._lru_put(key, vector)

    def get_or_put(self, key: str, vector: np.ndarray) -> np.ndarray:
        """Get from cache, or store and return the provided vector. Thread-safe."""
        with self._sync_lock:
            if key in self._lru:
                self._lru.move_to_end(key)
                self.stats_hits += 1
                return self._lru[key]
            self._lru_put(key, vector)
            self.stats_misses += 1
            return vector

    # ── Async API (for concurrent coroutine access) ─────────────────────

    async def read_vector(self, vector_id: int) -> np.ndarray:
        """Read a single vector by integer ID from the binary file.

        Checks LRU cache first, then deduplicates in-flight reads,
        then falls back to disk.
        """
        key = str(vector_id)

        # Step 1: LRU cache (instant)
        with self._sync_lock:
            if key in self._lru:
                self._lru.move_to_end(key)
                self.stats_hits += 1
                return self._lru[key]

        # Step 2: Deduplicate in-flight reads
        if key in self._in_flight:
            self.stats_deduped += 1
            return await asyncio.shield(self._in_flight[key])

        # Step 3: We're the one to read. Create a future for others to wait on.
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        self._in_flight[key] = future

        try:
            vector = await self._do_disk_read(vector_id)
            with self._sync_lock:
                self._lru_put(key, vector)
            future.set_result(vector)
            self.stats_misses += 1
            return vector
        except Exception as e:
            future.set_exception(e)
            raise
        finally:
            self._in_flight.pop(key, None)

    async def read_vectors_batch(self, vector_ids: List[int]) -> List[np.ndarray]:
        """Read multiple vectors, deduplicating both across the batch
        and across concurrent calls.

        Much faster than calling read_vector() in a loop — dedups
        within the batch too (same vector ID appearing multiple times
        in candidate lists is common with popular documents).
        """
        # Deduplicate requested IDs
        unique_ids = list(set(vector_ids))

        # Concurrent reads (each one deduplicates internally)
        tasks = [self.read_vector(vid) for vid in unique_ids]
        unique_results = await asyncio.gather(*tasks)

        # Reconstruct original order
        id_to_vec = dict(zip(unique_ids, unique_results))
        return [id_to_vec[vid] for vid in vector_ids]

    # ── Async doc-ID-keyed API (for scored_search dedup) ────────────────

    async def get_or_compute(self, doc_id: str, compute_fn) -> np.ndarray:
        """Get a cached vector by doc_id, or compute it via compute_fn.

        This deduplicates concurrent calls for the same doc_id:
        if 500 queries all want to score doc "wiki_42", only one
        actually calls compute_fn; the other 499 get the cached result.

        Args:
            doc_id: string document ID (e.g., "wiki_12345")
            compute_fn: async or sync callable that returns np.ndarray
        """
        # LRU check
        with self._sync_lock:
            if doc_id in self._lru:
                self._lru.move_to_end(doc_id)
                self.stats_hits += 1
                return self._lru[doc_id]

        # Dedup in-flight
        if doc_id in self._in_flight:
            self.stats_deduped += 1
            return await asyncio.shield(self._in_flight[doc_id])

        # We're the one to compute
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        self._in_flight[doc_id] = future

        try:
            # compute_fn may be sync or async
            if asyncio.iscoroutinefunction(compute_fn):
                vector = await compute_fn()
            else:
                vector = compute_fn()

            if vector is not None:
                with self._sync_lock:
                    self._lru_put(doc_id, vector)

            future.set_result(vector)
            self.stats_misses += 1
            return vector
        except Exception as e:
            future.set_exception(e)
            raise
        finally:
            self._in_flight.pop(doc_id, None)

    # ── Disk I/O ────────────────────────────────────────────────────────

    async def _do_disk_read(self, vector_id: int) -> np.ndarray:
        """Read a single vector from the binary file by integer offset.

        File format (DiskANN fbin): [uint32 N][uint32 D][float32 × N × D]
        Header is 8 bytes. Each vector is vector_bytes bytes.
        """
        if self.vector_file_path is None:
            raise RuntimeError("No vector file path configured for disk reads")

        # Use synchronous I/O in a thread pool to avoid blocking the event loop.
        # This is actually faster than aiofiles for random-access reads because
        # it avoids the aiofiles wrapper overhead and lets the OS page cache
        # handle the actual I/O scheduling.
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._sync_disk_read, vector_id)

    def _sync_disk_read(self, vector_id: int) -> np.ndarray:
        """Synchronous disk read — runs in a thread pool."""
        offset = 8 + vector_id * self.vector_bytes
        with open(self.vector_file_path, "rb") as f:
            f.seek(offset)
            raw = f.read(self.vector_bytes)
        if len(raw) < self.vector_bytes:
            raise IOError(
                f"Short read for vector {vector_id}: got {len(raw)} bytes, "
                f"expected {self.vector_bytes}"
            )
        return np.frombuffer(raw, dtype=np.float32).copy()

    # ── LRU internals ──────────────────────────────────────────────────

    def _lru_put(self, key: str, vector: np.ndarray) -> None:
        """Add to LRU, evicting least-recently-used if over capacity.
        Caller must hold self._sync_lock.
        """
        if key in self._lru:
            self._lru.move_to_end(key)
            return
        if len(self._lru) >= self._lru_capacity:
            self._lru.popitem(last=False)  # evict LRU
        self._lru[key] = vector

    # ── Stats ───────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Return cache statistics for monitoring."""
        total = self.stats_hits + self.stats_misses
        return {
            "cache_size": len(self._lru),
            "cache_capacity": self._lru_capacity,
            "hits": self.stats_hits,
            "misses": self.stats_misses,
            "deduped_reads": self.stats_deduped,
            "hit_rate": round(self.stats_hits / total, 4) if total > 0 else 0.0,
            "in_flight": len(self._in_flight),
        }

    async def close(self):
        """Cleanup."""
        self._lru.clear()
        self._in_flight.clear()
