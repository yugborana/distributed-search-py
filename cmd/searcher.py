"""
Shard Search Server

HTTP server that loads a Tantivy BM25 index.
BM25 handles keyword search. Cosine similarity for semantic scoring
is computed inline during scored_search (near-data scoring).

Mirrors: distributed-search/cmd/searcher/main.go

Usage:
    python cmd/searcher.py --port 8080 --index search.idx

Endpoints:
    GET /search?q=...&limit=20          — BM25 keyword search
    POST /scored_search                  — BM25 + semantic scoring at the shard
    GET /health                          — Health check
    GET /stats                           — Index statistics
"""

import argparse
import logging
import sys
import time
import asyncio
import os
import concurrent.futures
from pathlib import Path

import orjson

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import etcd3
except ImportError:
    etcd3 = None

from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse, ORJSONResponse
import uvicorn

from internal.index import SearchIndex

try:
    import grpc
    from internal.proto import shard_pb2, shard_pb2_grpc
except ImportError:
    grpc = None
    shard_pb2 = None
    shard_pb2_grpc = None

_grpc_server = None

if shard_pb2_grpc:
    class ShardServiceImpl(shard_pb2_grpc.ShardServiceServicer):
        async def Search(self, request, context):
            if not request.query or not index:
                return shard_pb2.SearchResponse(hits=[])
            loop = asyncio.get_running_loop()
            hits_dict = await loop.run_in_executor(_search_pool, index.search, request.query, request.limit)
            pb_hits = [
                shard_pb2.Hit(
                    id=str(h["id"]),
                    score=float(h["score"]),
                    bm25_score=float(h["score"]),
                    semantic_score=0.0,
                    title=str(h["title"]),
                    shard=shard_label
                )
                for h in hits_dict
            ]
            return shard_pb2.SearchResponse(hits=pb_hits)

        async def ScoredSearch(self, request, context):
            if not request.query or not index:
                return shard_pb2.SearchResponse(hits=[])
            loop = asyncio.get_running_loop()
            vec_list = list(request.query_vector) if request.query_vector else []
            hits_dict = await loop.run_in_executor(_search_pool, index.scored_search, request.query, vec_list, request.limit)
            pb_hits = [
                shard_pb2.Hit(
                    id=str(h["id"]),
                    score=float(h.get("bm25_score", h.get("score", 0.0))),
                    bm25_score=float(h.get("bm25_score", h.get("score", 0.0))),
                    semantic_score=float(h.get("semantic_score", 0.0)),
                    title=str(h["title"]),
                    shard=shard_label
                )
                for h in hits_dict
            ]
            return shard_pb2.SearchResponse(hits=pb_hits)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ── Globals (set during startup) ────────────────────────────────────────────
# PERF-PYDANTIC: ORJSONResponse bypasses FastAPI's jsonable_encoder() + stdlib
# json.dumps(). For scored_search responses with numeric scores, this is 3-10x faster.
app = FastAPI(title="Distributed Search — Shard Server", default_response_class=ORJSONResponse)
index: SearchIndex | None = None
shard_label: str = "single"
shard_id_global: int = -1
shard_port: int = 8080
shard_hostname: str = "localhost"
etcd_host: str = "localhost"

# PERF-01: Bounded thread pool for Tantivy FFI + numpy calls.
# Tantivy's C extension and numpy both release the GIL, so real parallelism
# happens. 4 workers = 4 concurrent searches per shard without blocking
# the event loop. The event loop stays free to accept new requests.
_search_pool = concurrent.futures.ThreadPoolExecutor(
    max_workers=4, thread_name_prefix="tantivy"
)


async def register_shard(shard_id: int, port: int, hostname: str, etcd_host: str):
    """Register this shard in etcd and keep the lease alive.
    
    PERF-12: All etcd operations (client creation, lease, put, refresh) are
    synchronous gRPC calls. They must run in a thread pool to avoid blocking
    the event loop.
    """
    if etcd3 is None:
        log.error("etcd3 module not installed. Cannot register shard.")
        return

    loop = asyncio.get_running_loop()

    # PERF-12: Initial registration — run all blocking etcd calls in executor
    def _register_sync():
        hosts = etcd_host.split(",")
        for host in hosts:
            try:
                client = etcd3.client(host=host.strip(), port=2379)
                lease = client.lease(ttl=30)
                key = f"/shards/active/{shard_id}"
                value = f"{hostname}:{port}"
                client.put(key, value, lease=lease)
                log.info(f"Registered shard in etcd ({host}): {key} -> {value}")
                return lease
            except Exception as e:
                log.warning(f"Failed to connect to ETCD at {host}: {e}")
                continue
        return None

    lease = await loop.run_in_executor(None, _register_sync)
    if lease is None:
        log.error("Failed to register with any etcd nodes.")
        return
        
    try:
        # Background keepalive — run blocking gRPC call in executor
        while True:
            await loop.run_in_executor(None, lease.refresh)
            await asyncio.sleep(10)
    except Exception as e:
        log.error(f"Keepalive failed: {e}")

async def refresh_index_task():
    """Background task to reload the index periodically to pick up new commits.
    
    PERF-12: All Tantivy FFI calls (reload, doc_count) must run in the thread
    pool to avoid blocking the event loop.
    """
    loop = asyncio.get_running_loop()
    while True:
        try:
            await asyncio.sleep(30)
            if index:
                old_count = await loop.run_in_executor(_search_pool, index.doc_count)
                await loop.run_in_executor(_search_pool, index.index.reload)
                new_count = await loop.run_in_executor(_search_pool, index.doc_count)
                if new_count != old_count:
                    log.info("Index auto-refreshed: %d -> %d docs", old_count, new_count)
        except Exception as e:
            log.error(f"Failed to refresh index: {e}")

@app.on_event("startup")
async def startup_event():
    global _grpc_server
    if shard_id_global >= 0:
        asyncio.create_task(register_shard(shard_id_global, shard_port, shard_hostname, etcd_host))
    asyncio.create_task(refresh_index_task())

    if grpc and shard_pb2_grpc:
        _grpc_server = grpc.aio.server()
        shard_pb2_grpc.add_ShardServiceServicer_to_server(ShardServiceImpl(), _grpc_server)
        grpc_port = shard_port + 1000
        _grpc_server.add_insecure_port(f"0.0.0.0:{grpc_port}")
        await _grpc_server.start()
        log.info(f"Shard gRPC server started on port {grpc_port}")

@app.on_event("shutdown")
async def shutdown_event():
    if _grpc_server:
        await _grpc_server.stop(None)


# ── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/search")
async def search_handler(
    q: str = Query(None, description="Search query"),
    limit: int = Query(20, ge=1, le=1000, description="Number of results"),
):
    """Execute a BM25 search query against the local Tantivy index."""
    if not q:
        return JSONResponse(
            status_code=400,
            content={"error": "missing 'q' parameter"},
        )

    start = time.time()
    # PERF-01: Run Tantivy BM25 search in thread pool to avoid blocking event loop
    loop = asyncio.get_running_loop()
    hits = await loop.run_in_executor(_search_pool, index.search, q, limit)
    took = time.time() - start

    return {
        "query": q,
        "total_hits": len(hits),
        "hits": hits,
        "took": f"{took*1000:.1f}ms",
        "shard": shard_label,
    }


@app.post("/scored_search")
async def scored_search_handler(request: Request):
    """Near-data scoring — BM25 + semantic scores computed at the shard.

    Accepts a JSON body with:
        q: str                     — search query text (for BM25)
        query_vector: List[float]  — query embedding (for cosine similarity)
        limit: int                 — number of results (default 100)

    Returns (doc_id, bm25_score, semantic_score, title) — NO vectors on the wire.
    """
    # PERF-PYDANTIC: Starlette's request.json() uses stdlib json.loads() internally.
    # orjson.loads() is 3-5x faster, especially for the 7KB body containing
    # a 384-dim float vector that arrives on every scored_search call.
    raw_body = await request.body()
    body = orjson.loads(raw_body)
    q = body.get("q")
    query_vector = body.get("query_vector")
    limit = body.get("limit", 100)

    if not q:
        return JSONResponse(
            status_code=400,
            content={"error": "missing 'q' in request body"},
        )

    start = time.time()
    # PERF-01: Run scored_search (Tantivy FFI + numpy BLAS) in thread pool.
    # Tantivy C extension releases the GIL during search, and numpy releases
    # the GIL during matrix ops — so real parallelism happens across threads.
    loop = asyncio.get_running_loop()
    hits = await loop.run_in_executor(
        _search_pool, index.scored_search, q, query_vector or [], limit
    )

    # Tag each hit with this shard's label
    for hit in hits:
        hit["shard"] = shard_label

    took = time.time() - start

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
    return "OK"


@app.get("/stats")
async def stats():
    """Return index stats (doc count, shard label).
    
    PERF-12: doc_count() is a Tantivy FFI call — run in thread pool.
    """
    doc_count = 0
    idx_path = None
    if index:
        loop = asyncio.get_running_loop()
        doc_count = await loop.run_in_executor(_search_pool, index.doc_count)
        idx_path = index.path
    return {
        "shard": shard_label,
        "doc_count": doc_count,
        "index_path": idx_path,
    }


# ── Startup ─────────────────────────────────────────────────────────────────

def main():
    global index, shard_label, shard_id_global, shard_port, shard_hostname, etcd_host

    parser = argparse.ArgumentParser(description="Shard search server")
    parser.add_argument("--port", type=int, default=8080, help="HTTP port")
    parser.add_argument("--index", default="search.idx", help="Base index path")
    parser.add_argument("--shard-id", type=int, default=-1,
                        help="Shard ID (-1 = single-node mode, no etcd)")
    parser.add_argument("--hostname", default="localhost",
                        help="Hostname for etcd registration")
    parser.add_argument("--etcd-host", default="localhost",
                        help="etcd server hostname")
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

    doc_count = index.doc_count()
    log.info(
        "Shard service ready :%d (index=%s, docs=%d)",
        args.port, index_path, doc_count,
    )

    # Start HTTP server
    # PERF-05: uvloop + httptools for the shard.
    # - uvloop: C-accelerated event loop, 2-4× faster socket I/O
    # - httptools: C-accelerated HTTP parser (llhttp), replaces h11
    # The shard can't use workers>1 (Tantivy index isn't fork-safe),
    # so faster per-connection I/O is critical. With PERF-01's ThreadPoolExecutor
    # handling the CPU work, the event loop only does socket I/O — making
    # uvloop + httptools the perfect complement.
    loop_policy = "auto"
    http_impl = "auto"
    try:
        import uvloop
        uvloop.install()
        loop_policy = "uvloop"
    except ImportError:
        pass
    try:
        import httptools  # noqa: F401
        http_impl = "httptools"
    except ImportError:
        pass

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=args.port,
        log_level="warning",
        access_log=False,
        loop=loop_policy,
        http=http_impl,
    )


if __name__ == "__main__":
    main()
