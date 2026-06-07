"""
Tantivy Search Index — Phase 1

Wraps tantivy-py to provide Bleve-equivalent functionality:
  - Schema definition (id, title, body)
  - Batch document indexing from JSONL
  - BM25 full-text search
  - Disk persistence (index directory)

Mirrors: distributed-search/internal/index/indexer.go

Key difference from Go/Bleve:
  - Bleve uses directory-based indexes → Tantivy also uses directory-based indexes
  - Bleve batch indexing → Tantivy writer with commit()
  - Both use BM25 scoring by default
"""

import json
import logging
import os
import time
import asyncio
from pathlib import Path

import tantivy

from internal.model import Doc
from internal.embed import OllamaClient
import orjson

log = logging.getLogger(__name__)


class SearchIndex:
    """Manages a Tantivy search index on disk.
    
    This is the Python equivalent of the Go `index.Indexer` struct.
    It wraps tantivy-py's Index, providing create/open/index/search operations.
    """

    def __init__(self, index_path: str, embed_client: OllamaClient = None):
        """Open an existing index or create a new one at the given path.
        
        Mirrors: index/indexer.go NewIndexer() lines 23-25 and NewIndexerWithVectors() lines 27-60
        
        Tantivy stores the index as a directory (just like Bleve stores
        search.bleve-N/ directories), so this is a direct equivalent.
        """
        self.path = index_path
        self.embed_client = embed_client
        self.schema = self._build_schema()

        if os.path.exists(index_path) and os.listdir(index_path):
            # Open existing index
            log.info("Opening existing index at %s", index_path)
            self.index = tantivy.Index(self.schema, path=index_path, reuse=True)
        else:
            # Create new index
            os.makedirs(index_path, exist_ok=True)
            log.info("Creating new index at %s", index_path)
            self.index = tantivy.Index(self.schema, path=index_path)

    @staticmethod
    def _build_schema() -> tantivy.Schema:
        """Define the document schema.
        
        Mirrors the Bleve index mapping from indexer.go lines 34-51:
          - id: text, stored (for retrieval)
          - title: text, stored (searchable + retrievable)
          - body: text, stored (searchable + retrievable, English stemming)
        
        Phase 5 will add: title_vector as bytes field for vector storage.
        """
        builder = tantivy.SchemaBuilder()

        # 'id' — stored for retrieval, indexed as raw (exact match)
        builder.add_text_field("id", stored=True, tokenizer_name="raw")

        # 'title' — full-text searchable with stemming, stored for display
        builder.add_text_field("title", stored=True, tokenizer_name="en_stem")

        # 'body' — full-text searchable with stemming, stored for retrieval
        builder.add_text_field("body", stored=True, tokenizer_name="en_stem")

        # 'title_vector' — stored for Phase 5 semantic search
        builder.add_bytes_field("title_vector", stored=True)

        return builder.build()

    def index_jsonl(self, jsonl_path: str, batch_size: int = 1000, max_docs: int = 0) -> int:
        """Read a JSONL file and index all documents into Tantivy.
        
        Mirrors: index/indexer.go IndexJSONL() lines 63-166
        
        The Go version uses Bleve batch operations. Tantivy-py doesn't have
        explicit batches, but we commit periodically for the same effect.
        
        Args:
            jsonl_path: Path to the JSONL file (one doc per line)
            batch_size: Number of docs to buffer before committing
            max_docs: Stop after this many docs (0 = no limit)
            
        Returns:
            Number of documents indexed
        """
        writer = self.index.writer(heap_size=128_000_000)  # 128MB writer heap

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
                    # Generate embedding if missing and client is available
                    try:
                        # Ensure we have an event loop and it's running
                        loop = asyncio.get_event_loop()
                        emb = loop.run_until_complete(self.embed_client.get_embedding(doc.title))
                        doc.title_vector = emb
                    except Exception as e:
                        log.warning("Embedding failed for doc %s: %s", doc.id, e)

                # Serialize vector to bytes
                vec_bytes = orjson.dumps(doc.title_vector) if doc.title_vector else b""

                # Add document to Tantivy
                writer.add_document(tantivy.Document(
                    id=[doc.id],
                    title=[doc.title],
                    body=[doc.body],
                    title_vector=[vec_bytes],
                ))
                indexed += 1

                # Periodic commit (equivalent to Bleve batch flush)
                if indexed % batch_size == 0:
                    writer.commit()

                    if time.time() - last_log > 2.0:
                        elapsed = time.time() - start
                        rate = indexed / elapsed if elapsed > 0 else 0
                        log.info(
                            "Progress: %d docs indexed (%.0f/sec)",
                            indexed, rate,
                        )
                        last_log = time.time()

                if max_docs > 0 and indexed >= max_docs:
                    log.info("Reached max-docs limit (%d)", max_docs)
                    break

        # Final commit for remaining docs
        writer.commit()

        # Reload to make new documents searchable
        self.index.reload()

        elapsed = time.time() - start
        rate = indexed / elapsed if elapsed > 0 else 0
        log.info("Indexing complete:")
        log.info("  %d docs indexed, %d skipped", indexed, skipped)
        log.info("  %.1fs elapsed (%.0f docs/sec)", elapsed, rate)

        return indexed

    def search(self, query_str: str, limit: int = 20) -> list[dict]:
        """Execute a BM25 search query against the index.
        
        Mirrors: searcher/main.go searchHandler() lines 165-208
        
        Phase 3: No longer returns title_vector in responses.
        Scoring happens at the shard via scored_search() instead.
        
        Args:
            query_str: The user's search query text
            limit: Maximum number of results to return
            
        Returns:
            List of hit dicts with 'id', 'score', 'title' keys
        """
        searcher = self.index.searcher()

        # Parse query against title and body fields (BM25 scoring)
        query = self.index.parse_query(query_str, ["title", "body"])

        # Execute search — returns (score, doc_address) tuples
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

    def scored_search(self, query_str: str, query_vector: list[float], limit: int = 20,
                       dedup_cache=None) -> list[dict]:
        """Near-data scoring: BM25 + semantic scoring done locally at the shard.
        
        Phase 3: Computes both BM25 and cosine similarity scores right here,
        returning only (doc_id, bm25_score, semantic_score, title).
        
        Phase 4: When dedup_cache (IODedupReader) is provided, parsed doc
        vectors are cached in its LRU. Under concurrent queries, popular
        docs' vectors are parsed from Tantivy only once — subsequent
        queries hit the cache. Under Zipf load, the top 1% of docs appear
        in ~50% of queries, so this avoids massive redundant parsing.
        
        Args:
            query_str: The user's search query text (for BM25)
            query_vector: The query embedding (for cosine similarity)
            limit: Maximum number of results to return
            dedup_cache: Optional IODedupReader for vector caching (Phase 4)
            
        Returns:
            List of hit dicts with 'id', 'bm25_score', 'semantic_score', 'title'
        """
        import numpy as np

        searcher = self.index.searcher()
        query = self.index.parse_query(query_str, ["title", "body"])
        search_result = searcher.search(query, limit)

        # Pre-compute query vector norm for cosine similarity
        q_vec = None
        q_norm = 0.0
        if query_vector and len(query_vector) > 0:
            q_vec = np.array(query_vector, dtype=np.float32)
            q_norm = float(np.linalg.norm(q_vec))

        hits = []
        for bm25_score, doc_address in search_result.hits:
            doc = searcher.doc(doc_address)
            doc_id = doc["id"][0]
            title = doc["title"][0]

            # Compute semantic score locally (Phase 3 near-data scoring)
            semantic_score = 0.0
            if q_vec is not None and q_norm > 0:
                try:
                    doc_vec = None

                    # Phase 4: check dedup cache first
                    if dedup_cache is not None:
                        doc_vec = dedup_cache.get_cached(doc_id)

                    # Cache miss — parse from Tantivy stored field
                    if doc_vec is None:
                        vec_bytes = doc.get("title_vector", [None])
                        if vec_bytes and vec_bytes[0]:
                            doc_vec = np.array(orjson.loads(vec_bytes[0]), dtype=np.float32)
                            # Phase 4: store in dedup cache for future queries
                            if dedup_cache is not None:
                                dedup_cache.put(doc_id, doc_vec)

                    # Cosine similarity
                    if doc_vec is not None:
                        doc_norm = float(np.linalg.norm(doc_vec))
                        if doc_norm > 0:
                            semantic_score = float(np.dot(q_vec, doc_vec) / (q_norm * doc_norm))
                except Exception:
                    pass

            hits.append({
                "id": doc_id,
                "bm25_score": round(float(bm25_score), 6),
                "semantic_score": round(semantic_score, 6),
                "title": title,
            })

        return hits

    def doc_count(self) -> int:
        """Return the total number of documents in the index."""
        searcher = self.index.searcher()
        return searcher.num_docs

    def close(self):
        """No-op for Tantivy (index is flushed on commit).
        
        Included for API parity with the Go Indexer.Close().
        """
        pass
