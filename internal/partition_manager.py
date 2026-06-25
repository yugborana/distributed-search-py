"""
Partition Manager — In-Process Multi-Shard Search

Manages multiple Tantivy SearchIndex partitions within a single process.
Each partition is queried in parallel via a ThreadPoolExecutor. Because
Tantivy's C extension releases the GIL during FFI calls, this achieves
true CPU parallelism identical to separate processes — but with zero
network/serialization overhead.

Usage:
    pm = PartitionManager("/app/search.idx", num_partitions=8)
    hits = await pm.search_all("algorithm", limit=20)
    scored = await pm.scored_search_all("algorithm", query_vec, limit=20)
"""

import asyncio
import concurrent.futures
import logging
import os
from typing import List, Dict, Any, Optional

import numpy as np

from internal.index import SearchIndex

log = logging.getLogger(__name__)


class PartitionManager:
    """Manages N Tantivy index partitions in-process with thread-parallel search."""

    def __init__(self, index_base: str, num_partitions: int = 8, max_workers: int = 8):
        """
        Load all partition indexes.

        Args:
            index_base: Base path for indexes (e.g. "/app/search.idx").
                        Partition i lives at "{index_base}-{i}".
            num_partitions: Number of partitions to load.
            max_workers: Thread pool size. Should equal num_partitions for
                         maximum GIL-released parallelism.
        """
        self.num_partitions = num_partitions
        self.partitions: List[SearchIndex] = []
        self._pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="partition"
        )

        total_docs = 0
        for i in range(num_partitions):
            path = f"{index_base}-{i}"
            if not os.path.exists(path):
                log.warning("Partition path %s does not exist, skipping", path)
                continue
            idx = SearchIndex(path)
            doc_count = idx.doc_count()
            total_docs += doc_count
            self.partitions.append(idx)
            log.info("Loaded partition %d: %s (%d docs)", i, path, doc_count)

        log.info(
            "PartitionManager ready: %d/%d partitions loaded, %d total docs",
            len(self.partitions), num_partitions, total_docs,
        )

    async def search_all(self, query: str, limit: int = 20) -> Dict[str, Any]:
        """Search all partitions in parallel and merge results by BM25 score.

        Returns a tuple of (sorted_hits, partitions_responded).
        """
        loop = asyncio.get_running_loop()

        # Dispatch one search per partition into the thread pool
        futures = [
            loop.run_in_executor(self._pool, idx.search, query, limit)
            for idx in self.partitions
        ]
        partition_results = await asyncio.gather(*futures, return_exceptions=True)

        all_hits: List[Dict[str, Any]] = []
        partitions_responded = 0
        for i, result in enumerate(partition_results):
            if isinstance(result, Exception):
                log.warning("Partition %d search failed: %s", i, result)
                continue
            if result:
                partitions_responded += 1
                for hit in result:
                    hit["shard"] = f"partition-{i}"
                    all_hits.append(hit)

        # Merge: sort by BM25 score descending, take top limit
        all_hits.sort(key=lambda h: h.get("score", 0), reverse=True)
        return all_hits[:limit], partitions_responded

    async def scored_search_all(
        self, query: str, query_vector: List[float], limit: int = 20
    ) -> Dict[str, Any]:
        """Scored search all partitions in parallel and merge results.

        PERF-19: Pre-normalizes the query vector ONCE here instead of
        letting each of the 8 partition threads do it independently.
        Saves 8x redundant np.asarray() + np.linalg.norm() calls.
        """
        loop = asyncio.get_running_loop()

        # PERF-19: Pre-normalize query vector once, share across all partitions.
        # Without this, each of the 8 threads independently converts the Python
        # list to numpy, computes the norm, and normalizes — 8x wasted work.
        pre_normalized_vec = None
        if query_vector is not None and len(query_vector) > 0:
            if isinstance(query_vector, np.ndarray):
                q_vec = query_vector.astype(np.float32, copy=False)
            else:
                q_vec = np.asarray(query_vector, dtype=np.float32)
            q_norm = np.linalg.norm(q_vec)
            if q_norm > 0:
                q_vec *= (1.0 / q_norm)
                pre_normalized_vec = q_vec

        # Pass numpy array directly — scored_search will detect it's already
        # normalized and skip redundant conversion
        vec_arg = pre_normalized_vec if pre_normalized_vec is not None else query_vector

        futures = [
            loop.run_in_executor(
                self._pool, idx.scored_search, query, vec_arg, limit
            )
            for idx in self.partitions
        ]
        partition_results = await asyncio.gather(*futures, return_exceptions=True)

        all_hits: List[Dict[str, Any]] = []
        partitions_responded = 0
        for i, result in enumerate(partition_results):
            if isinstance(result, Exception):
                log.warning("Partition %d scored_search failed: %s", i, result)
                continue
            if result:
                partitions_responded += 1
                for hit in result:
                    hit["shard"] = f"partition-{i}"
                    all_hits.append(hit)

        return all_hits, partitions_responded

    def total_docs(self) -> int:
        """Return aggregate doc count across all partitions."""
        return sum(idx.doc_count() for idx in self.partitions)

    def reload_all(self):
        """Reload all partition indexes (picks up new commits)."""
        for i, idx in enumerate(self.partitions):
            try:
                idx.index.reload()
            except Exception as e:
                log.warning("Failed to reload partition %d: %s", i, e)
