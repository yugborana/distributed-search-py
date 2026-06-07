"""
Shard Search Server — Phase 1 + Quantization Cascade

HTTP server that loads a Tantivy BM25 index AND a FAISS IVF-PQ vector index.
The BM25 index handles keyword search. The vector index handles semantic search.
The coordinator fuses both result sets via RRF or weighted fusion.

Mirrors: distributed-search/cmd/searcher/main.go

Usage:
    python cmd/searcher.py --port 8080 --index search.idx

Endpoints:
    GET /search?q=...&limit=20          — BM25 keyword search
    POST /vector_search                  — Vector similarity search (accepts query vector)
    GET /health                          — Health check
    GET /stats                           — Index statistics
"""

import argparse
import logging
import sys
import time
import asyncio
import os
from pathlib import Path

import numpy as np
import orjson

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import etcd3
except ImportError:
    etcd3 = None

from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse
import uvicorn

from internal.index import SearchIndex
from internal.search.index.index_factory import create_vector_index
from internal.search.index import BaseVectorIndex
from internal.search.io.dedup_reader import IODedupReader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ── Globals (set during startup) ────────────────────────────────────────────
app = FastAPI(title="Distributed Search — Shard Server")
index: SearchIndex | None = None
vector_index: BaseVectorIndex | None = None  # Phase 1: FAISS IVF-PQ
dedup_reader: IODedupReader | None = None    # Phase 4: I/O deduplication
shard_label: str = "single"
shard_id_global: int = -1
shard_port: int = 8080
shard_hostname: str = "localhost"
etcd_host: str = "localhost"


async def register_shard(shard_id: int, port: int, hostname: str, etcd_host: str):
    if etcd3 is None:
        log.error("etcd3 module not installed. Cannot register shard.")
        return
        
    hosts = etcd_host.split(",")
    client = None
    lease = None
    for host in hosts:
        try:
            client = etcd3.client(host=host.strip(), port=2379)
            lease = client.lease(ttl=30)
            key = f"/shards/active/{shard_id}"
            value = f"{hostname}:{port}"
            client.put(key, value, lease=lease)
            log.info(f"Registered shard in etcd ({host}): {key} -> {value}")
            break
        except Exception as e:
            log.warning(f"Failed to connect to ETCD at {host}: {e}")
            client = None
            continue
            
    if not client:
        log.error("Failed to register with any etcd nodes.")
        return
        
    try:
        # Background keepalive
        while True:
            lease.refresh()
            await asyncio.sleep(10)
    except Exception as e:
        log.error(f"Keepalive failed: {e}")

async def refresh_index_task():
    """Background task to reload the index periodically to pick up new commits."""
    while True:
        try:
            await asyncio.sleep(30)
            if index:
                old_count = index.doc_count()
                index.index.reload()
                new_count = index.doc_count()
                if new_count != old_count:
                    log.info("Index auto-refreshed: %d -> %d docs", old_count, new_count)
        except Exception as e:
            log.error(f"Failed to refresh index: {e}")

@app.on_event("startup")
async def startup_event():
    if shard_id_global >= 0:
        asyncio.create_task(register_shard(shard_id_global, shard_port, shard_hostname, etcd_host))
    asyncio.create_task(refresh_index_task())
    # Build vector index in background to avoid blocking startup
    if vector_index is not None and index is not None:
        asyncio.get_event_loop().run_in_executor(None, _build_vector_index_from_tantivy)
    # Phase 4: Log dedup reader status
    if dedup_reader is not None:
        log.info("IODedupReader active: cache capacity=%d vectors", dedup_reader._lru_capacity)


# ── Endpoints ───────────────────────────────────────────────────────────────
# Mirrors: searcher/main.go searchHandler() lines 165-208

@app.get("/search")
async def search_handler(
    q: str = Query(None, description="Search query"),
    limit: int = Query(20, ge=1, le=1000, description="Number of results"),
):
    """Execute a BM25 search query against the local Tantivy index.
    
    This is the exact equivalent of the Go searchHandler():
    1. Parse query string and limit from URL params
    2. Build a Tantivy query against title + body fields
    3. Execute search (BM25 scoring, automatic)
    4. Return JSON with hits
    """
    if not q:
        return JSONResponse(
            status_code=400,
            content={"error": "missing 'q' parameter"},
        )

    start = time.time()

    hits = index.search(q, limit)

    took = time.time() - start
    log.info("'%s' → %d hits in %.3fs (shard=%s)", q, len(hits), took, shard_label)

    return {
        "query": q,
        "total_hits": len(hits),
        "hits": hits,
        "took": f"{took*1000:.1f}ms",
        "shard": shard_label,
    }


@app.post("/scored_search")
async def scored_search_handler(request: Request):
    """Phase 3: Near-data scoring — BM25 + semantic scores computed at the shard.

    Accepts a JSON body with:
        q: str                     — search query text (for BM25)
        query_vector: List[float]  — query embedding (for cosine similarity)
        limit: int                 — number of results (default 100)

    Returns (doc_id, bm25_score, semantic_score, title) — NO vectors on the wire.

    Network savings vs returning vectors:
        Before: 100 hits × 1,536 bytes/vector = 153,600 bytes per shard
        After:  100 hits × 8 bytes/scores     =     800 bytes per shard
        = 192× reduction in network traffic
    """
    body = await request.json()
    q = body.get("q")
    query_vector = body.get("query_vector")
    limit = body.get("limit", 100)

    if not q:
        return JSONResponse(
            status_code=400,
            content={"error": "missing 'q' in request body"},
        )

    start = time.time()

    # Phase 4: scored_search with I/O dedup cache
    # The dedup_reader caches parsed title_vectors across concurrent queries
    # so the same popular doc's vector is parsed from Tantivy only once.
    hits = index.scored_search(q, query_vector or [], limit, dedup_cache=dedup_reader)

    # Tag each hit with this shard's label
    for hit in hits:
        hit["shard"] = shard_label

    took = time.time() - start
    log.info("scored_search '%s' → %d hits in %.3fs (shard=%s)", q, len(hits), took, shard_label)

    return {
        "total_hits": len(hits),
        "hits": hits,
        "took": f"{took*1000:.1f}ms",
        "shard": shard_label,
        "shard_id": shard_id_global,
        "search_time_ms": round(took * 1000, 2),
    }


@app.get("/health")
async def health():
    """Health check endpoint.
    
    Mirrors: searcher/main.go health handler, lines 82-85
    """
    return "OK"


@app.post("/vector_search")
async def vector_search_handler(request: Request):
    """Vector similarity search using the FAISS IVF-PQ index.

    Accepts a JSON body with:
        query_vector: List[float]  — the query embedding (384 dims)
        k: int                     — number of results (default 20)

    Returns top-k (doc_id, similarity_score) pairs.
    This endpoint is called by the coordinator for hybrid search fusion.
    """
    if vector_index is None or vector_index.size == 0:
        return JSONResponse(
            status_code=503,
            content={"error": "vector index not ready", "shard": shard_label},
        )

    body = await request.json()
    query_vector = body.get("query_vector")
    k = body.get("k", 20)

    if not query_vector:
        return JSONResponse(
            status_code=400,
            content={"error": "missing 'query_vector' in request body"},
        )

    start = time.time()
    q = np.array(query_vector, dtype=np.float32)
    results = vector_index.search(q, k)
    took = time.time() - start

    hits = [
        {"id": doc_id, "score": round(score, 6), "shard": shard_label}
        for doc_id, score in results
    ]

    log.info("vector_search → %d hits in %.3fs (shard=%s)", len(hits), took, shard_label)

    return {
        "total_hits": len(hits),
        "hits": hits,
        "took": f"{took*1000:.1f}ms",
        "shard": shard_label,
        "index_type": type(vector_index).__name__,
    }


@app.get("/stats")
async def stats():
    """Return index stats (doc count, shard label, vector index info, dedup cache)."""
    return {
        "shard": shard_label,
        "doc_count": index.doc_count() if index else 0,
        "index_path": index.path if index else None,
        "vector_index": {
            "type": type(vector_index).__name__ if vector_index else None,
            "size": vector_index.size if vector_index else 0,
        },
        "dedup_cache": dedup_reader.get_stats() if dedup_reader else None,
    }


# ── Vector Index Bootstrap ──────────────────────────────────────────────────

def _build_vector_index_from_tantivy():
    """Extract vectors from Tantivy and build the FAISS IVF-PQ index.

    This runs once at startup (in a background thread). It:
    1. Checks if a saved vector index exists on disk → loads it
    2. Otherwise, scans all documents in Tantivy, extracts title_vector,
       and calls vector_index.build()
    3. Saves the built index to disk for fast restarts

    In our codebase, vectors are stored inside Tantivy as serialised bytes.
    This function bridges the gap between Tantivy storage and FAISS indexing.
    """
    global vector_index
    if vector_index is None or index is None:
        return

    vec_index_path = f"{index.path}.vecidx"

    # Try loading from disk first (fast restart)
    if Path(vec_index_path).exists():
        try:
            vector_index.load(vec_index_path)
            log.info("Loaded vector index from disk: %d vectors", vector_index.size)
            return
        except Exception as e:
            log.warning("Failed to load saved vector index: %s — rebuilding", e)

    # Extract vectors from Tantivy
    log.info("Extracting vectors from Tantivy index for FAISS build...")
    searcher = index.index.searcher()
    num_docs = searcher.num_docs

    if num_docs == 0:
        log.warning("No documents in Tantivy — skipping vector index build")
        return

    vectors = []
    doc_ids = []
    # Scan all docs by searching with a match-all pattern
    # Tantivy doesn't have a direct "iterate all docs" API,
    # so we do a broad search with a high limit
    try:
        all_query = index.index.parse_query("*", ["title"])
        all_results = searcher.search(all_query, num_docs)

        for _score, doc_addr in all_results.hits:
            doc = searcher.doc(doc_addr)
            doc_id = doc["id"][0]

            # Extract stored vector bytes
            try:
                vec_bytes = doc["title_vector"]
                if vec_bytes and vec_bytes[0]:
                    vec = orjson.loads(vec_bytes[0])
                    if vec and len(vec) > 0:
                        vectors.append(vec)
                        doc_ids.append(doc_id)
            except Exception:
                continue

    except Exception as e:
        log.error("Failed to scan Tantivy for vectors: %s", e)
        return

    if len(vectors) == 0:
        log.warning("No vectors found in Tantivy index — vector search disabled")
        return

    log.info("Extracted %d vectors from %d docs — building FAISS index...",
             len(vectors), num_docs)

    vec_array = np.array(vectors, dtype=np.float32)
    vector_index.build(vec_array, doc_ids)

    # Persist to disk
    try:
        vector_index.save(vec_index_path)
        log.info("Vector index saved to %s", vec_index_path)
    except Exception as e:
        log.warning("Failed to save vector index: %s", e)


# ── Startup ─────────────────────────────────────────────────────────────────

def main():
    global index, vector_index, shard_label, shard_id_global, shard_port, shard_hostname, etcd_host

    parser = argparse.ArgumentParser(description="Shard search server")
    parser.add_argument("--port", type=int, default=8080, help="HTTP port")
    parser.add_argument("--index", default="search.idx", help="Base index path")
    parser.add_argument("--shard-id", type=int, default=-1,
                        help="Shard ID (-1 = single-node mode, no etcd)")
    parser.add_argument("--hostname", default="localhost",
                        help="Hostname for etcd registration (Phase 2)")
    parser.add_argument("--etcd-host", default="localhost",
                        help="etcd server hostname")
    parser.add_argument("--config", default="system_config.yaml",
                        help="Path to system_config.yaml")
    args = parser.parse_args()

    # Resolve index path (mirrors Go: fmt.Sprintf("%s-%d", indexBase, shardID))
    index_path = args.index
    if args.shard_id >= 0:
        index_path = f"{args.index}-{args.shard_id}"
        shard_label = f"shard-{args.shard_id}"
        shard_id_global = args.shard_id
        shard_port = args.port
        shard_hostname = args.hostname
        etcd_host = args.etcd_host
    else:
        shard_label = "single"

    # Load the Tantivy BM25 index
    log.info("Loading BM25 index from %s ...", index_path)
    index = SearchIndex(index_path)

    # Initialize the vector index from config
    config_path = args.config
    if not Path(config_path).exists():
        # Try inside /app (Docker) or project root
        for candidate in ["/app/system_config.yaml", "system_config.yaml"]:
            if Path(candidate).exists():
                config_path = candidate
                break

    try:
        vector_index = create_vector_index(config_path, shard_id=args.shard_id)
        log.info("Vector index initialized: %s", type(vector_index).__name__)
    except Exception as e:
        log.warning("Vector index disabled: %s", e)
        vector_index = None

    # Phase 4: Initialize I/O deduplication reader
    # Caches parsed doc vectors across concurrent queries.
    # 512MB ≈ 349K cached 384-dim vectors — covers the popular tail.
    dedup_reader = IODedupReader(
        vector_file_path=None,  # doc-ID-keyed mode (vectors come from Tantivy, not a flat file)
        vector_dim=384,
        cache_size_mb=512,
    )
    log.info("IODedupReader initialized: %dMB LRU cache", 512)

    doc_count = index.doc_count()
    log.info(
        "Shard service ready :%d (index=%s, docs=%d, vector=%s, dedup=ON)",
        args.port, index_path, doc_count,
        type(vector_index).__name__ if vector_index else "disabled",
    )

    # Start HTTP server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
