"""
Tantivy Search Index

Wraps tantivy-py to provide:
  - Schema definition (id, title, body, title_vector)
  - Batch document indexing from JSONL
  - BM25 full-text search
  - Near-data scored search (BM25 + batched cosine similarity)
  - Disk persistence (index directory)

Mirrors: distributed-search/internal/index/indexer.go
"""

import json
import logging
import os
import time
import asyncio
import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np
import tantivy

from internal.model import Doc
import orjson

log = logging.getLogger(__name__)

# Simple LRU cache for parsed doc vectors (replaces the heavyweight IODedupReader)
# PERF-01 safety: ThreadPoolExecutor runs scored_search on multiple threads,
# so all cache access must be protected by a lock.
_VEC_CACHE_MAX = 50_000  # fits ~73MB for 384-dim float32 vectors
_vec_cache: OrderedDict[str, np.ndarray] = OrderedDict()
_vec_cache_lock = threading.Lock()


def _vec_cache_get(key: str) -> np.ndarray | None:
    """Get a cached vector, or None. Moves to end (most-recently-used)."""
    with _vec_cache_lock:
        if key in _vec_cache:
            _vec_cache.move_to_end(key)
            return _vec_cache[key]
    return None


def _vec_cache_put(key: str, vec: np.ndarray):
    """Put a vector into the LRU cache."""
    with _vec_cache_lock:
        if key in _vec_cache:
            _vec_cache.move_to_end(key)
            return
        if len(_vec_cache) >= _VEC_CACHE_MAX:
            _vec_cache.popitem(last=False)
        _vec_cache[key] = vec


class SearchIndex:
    """Manages a Tantivy search index on disk.
    
    This is the Python equivalent of the Go `index.Indexer` struct.
    """

    def __init__(self, index_path: str, embed_client=None):
        """Open an existing index or create a new one at the given path."""
        self.path = index_path
        self.embed_client = embed_client
        self.schema = self._build_schema()

        if os.path.exists(index_path) and os.listdir(index_path):
            log.info("Opening existing index at %s", index_path)
            self.index = tantivy.Index(self.schema, path=index_path, reuse=True)
        else:
            os.makedirs(index_path, exist_ok=True)
            log.info("Creating new index at %s", index_path)
            self.index = tantivy.Index(self.schema, path=index_path)

    @staticmethod
    def _build_schema() -> tantivy.Schema:
        """Define the document schema."""
        builder = tantivy.SchemaBuilder()
        builder.add_text_field("id", stored=True, tokenizer_name="raw")
        builder.add_text_field("title", stored=True, tokenizer_name="en_stem")
        builder.add_text_field("body", stored=True, tokenizer_name="en_stem")
        builder.add_bytes_field("title_vector", stored=True)
        return builder.build()

    def index_jsonl(self, jsonl_path: str, batch_size: int = 1000, max_docs: int = 0) -> int:
        """Read a JSONL file and index all documents into Tantivy."""
        writer = self.index.writer(heap_size=128_000_000)

        indexed = 0
        skipped = 0
        start = time.time()
        last_log = time.time()

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    log.warning("Skip bad JSON line: %s", e)
                    skipped += 1
                    continue

                doc = Doc.from_dict(data)

                if self.embed_client and not doc.title_vector:
                    try:
                        loop = asyncio.get_event_loop()
                        emb = loop.run_until_complete(self.embed_client.get_embedding(doc.title))
                        doc.title_vector = emb
                    except Exception as e:
                        log.warning("Embedding failed for doc %s: %s", doc.id, e)

                # PERF-08: Store vectors as raw float32 bytes instead of JSON.
                # Raw bytes: np.frombuffer() recovery is ~0.001ms (zero-copy)
                # JSON bytes: orjson.loads() + np.array() is ~0.08ms (parse + copy)
                if doc.title_vector:
                    vec_bytes = np.array(doc.title_vector, dtype=np.float32).tobytes()
                else:
                    vec_bytes = b""

                writer.add_document(tantivy.Document(
                    id=[doc.id],
                    title=[doc.title],
                    body=[doc.body],
                    title_vector=[vec_bytes],
                ))
                indexed += 1

                if indexed % batch_size == 0:
                    writer.commit()
                    if time.time() - last_log > 2.0:
                        elapsed = time.time() - start
                        rate = indexed / elapsed if elapsed > 0 else 0
                        log.info("Progress: %d docs indexed (%.0f/sec)", indexed, rate)
                        last_log = time.time()

                if max_docs > 0 and indexed >= max_docs:
                    log.info("Reached max-docs limit (%d)", max_docs)
                    break

        writer.commit()
        self.index.reload()

        elapsed = time.time() - start
        rate = indexed / elapsed if elapsed > 0 else 0
        log.info("Indexing complete: %d docs indexed, %d skipped in %.1fs (%.0f docs/sec)",
                 indexed, skipped, elapsed, rate)
        return indexed

    def search(self, query_str: str, limit: int = 20) -> list[dict]:
        """Execute a BM25 search query against the index."""
        searcher = self.index.searcher()
        query = self.index.parse_query(query_str, ["title", "body"])
        search_result = searcher.search(query, limit)

        hits = []
        for score, doc_address in search_result.hits:
            doc = searcher.doc(doc_address)
            hits.append({
                "id": doc["id"][0],
                "score": round(score, 6),
                "title": doc["title"][0],
            })
        return hits

    def scored_search(self, query_str: str, query_vector: list[float],
                      limit: int = 20) -> list[dict]:
        """Near-data scoring: BM25 + batched cosine similarity at the shard.
        
        PERF-07 + PERF-08 optimizations:
          - Query vector: np.asarray() with contiguous fast-path
          - Doc vectors: np.frombuffer() from raw float32 bytes (zero-copy)
          - Fallback to orjson.loads() for legacy JSON-encoded vectors
          - Batched cosine via single BLAS matmul
        """
        searcher = self.index.searcher()
        query = self.index.parse_query(query_str, ["title", "body"])
        search_result = searcher.search(query, limit)

        # PERF-07/19: If query_vector is already a pre-normalized numpy array
        # (from PartitionManager), skip the redundant conversion and normalization.
        q_vec = None
        if query_vector is not None and len(query_vector) > 0:
            if isinstance(query_vector, np.ndarray):
                # Already a numpy array — likely pre-normalized by PartitionManager
                q_vec = query_vector if query_vector.dtype == np.float32 else query_vector.astype(np.float32)
            else:
                q_vec = np.asarray(query_vector, dtype=np.float32)
                q_norm = np.linalg.norm(q_vec)
                if q_norm > 0:
                    q_vec *= (1.0 / q_norm)  # in-place normalize (avoids alloc)
                else:
                    q_vec = None

        # Phase 1: Extract all BM25 hits and their vectors in one pass
        n_hits = len(search_result.hits)
        doc_ids = [None] * n_hits
        titles = [None] * n_hits
        bm25_scores = [0.0] * n_hits
        doc_vecs = [None] * n_hits
        has_vec_flags = [False] * n_hits

        for i, (bm25_score, doc_address) in enumerate(search_result.hits):
            doc = searcher.doc(doc_address)
            doc_id = doc["id"][0]

            doc_ids[i] = doc_id
            titles[i] = doc["title"][0]
            bm25_scores[i] = float(bm25_score)

            # Try to get the doc vector (from cache or Tantivy)
            if q_vec is not None:
                doc_vec = _vec_cache_get(doc_id)
                if doc_vec is None:
                    try:
                        vec_bytes_list = doc["title_vector"]
                        if vec_bytes_list and len(vec_bytes_list) > 0 and vec_bytes_list[0] is not None:
                            raw = vec_bytes_list[0]
                            # PERF-08: Detect format — raw float32 bytes vs JSON
                            # Raw format: len is exact multiple of 4 and NOT valid JSON
                            # (JSON always starts with '[' = 0x5B, float32 won't)
                            if isinstance(raw, bytes) and len(raw) > 0 and raw[0:1] != b'[':
                                # Raw float32 bytes — np.frombuffer is ~zero-copy
                                doc_vec = np.frombuffer(raw, dtype=np.float32).copy()
                            else:
                                # Legacy JSON format — fallback
                                if isinstance(raw, memoryview):
                                    raw = bytes(raw)
                                doc_vec = np.asarray(orjson.loads(raw), dtype=np.float32)
                            
                            if len(doc_vec) > 0:
                                dnorm = np.linalg.norm(doc_vec)
                                if dnorm > 0:
                                    doc_vec *= (1.0 / dnorm)  # in-place normalize
                                    _vec_cache_put(doc_id, doc_vec)
                                else:
                                    doc_vec = None
                    except Exception:
                        doc_vec = None

                if doc_vec is not None:
                    doc_vecs[i] = doc_vec
                    has_vec_flags[i] = True

        # Phase 2: Batched cosine similarity — single matrix multiply
        semantic_scores = np.zeros(n_hits, dtype=np.float32)
        if q_vec is not None and any(has_vec_flags):
            # Build matrix of only the docs that have vectors
            vec_indices = [i for i in range(n_hits) if has_vec_flags[i]]
            if vec_indices:
                vec_matrix = np.stack([doc_vecs[i] for i in vec_indices])  # (M, D)
                cos_scores = vec_matrix @ q_vec  # single BLAS call → (M,)
                for j, idx in enumerate(vec_indices):
                    semantic_scores[idx] = cos_scores[j]

        # Phase 3: Build response — pre-allocated list
        hits = [None] * n_hits
        for i in range(n_hits):
            hits[i] = {
                "id": doc_ids[i],
                "bm25_score": round(bm25_scores[i], 6),
                "semantic_score": round(float(semantic_scores[i]), 6),
                "title": titles[i],
            }

        return hits

    def doc_count(self) -> int:
        """Return the total number of documents in the index."""
        searcher = self.index.searcher()
        return searcher.num_docs

    def close(self):
        """No-op for Tantivy (index is flushed on commit)."""
        pass
