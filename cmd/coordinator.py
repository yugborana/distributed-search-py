"""
Coordinator — Phase 2

Discovers shards via etcd and fans out search queries to all of them,
merging the results.
"""

import argparse
import asyncio
import orjson
import logging
import time
import hashlib
import gzip
import uvloop
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Dict, Any

from internal.embed import OllamaClient
from internal.hybrid import fuse_with_weights, fuse_with_rrf
from internal.routing import HotTermMapper

from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse
import uvicorn
import httpx
try:
    import etcd3
except ImportError:
    etcd3 = None

try:
    import redis.asyncio as aioredis
except ImportError:
    aioredis = None

log = logging.getLogger("coordinator")
log.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
log.addHandler(handler)

app = FastAPI(title="Distributed Search — Coordinator")

etcd_host_global = os.environ.get("ETCD_HOSTS", "localhost")
redis_host_global = os.environ.get("REDIS_HOST", "localhost")
ollama_host_global = os.environ.get("OLLAMA_HOST", "localhost")
fusion_alpha_default = float(os.environ.get("FUSION_ALPHA", "0.5"))

rdb = None
embed_client = None
routing_mapper = None
http_pool: httpx.AsyncClient = None  # Shared connection pool (Fix 2)
CACHE_TTL = 300
HYBRID_CACHE_TTL = 60
SHARD_TIMEOUT = 5.0  # Per-shard timeout in seconds (Fix 3)

async def hot_term_refresh_task():
    """Background task to keep hot terms synced from etcd."""
    while True:
        if routing_mapper:
            routing_mapper.refresh()
        await asyncio.sleep(30)

@app.on_event("startup")
async def startup_event():
    global rdb, embed_client, http_pool
    
    if aioredis is not None:
        # Note: decode_responses=False is important here as we store raw pickled/binary data
        rdb = aioredis.from_url(f"redis://{redis_host_global}:6379", decode_responses=False)

    # Initialize Ollama Client with Redis for vector caching (Fix 1)
    embed_client = OllamaClient(
        base_url=f"http://{ollama_host_global}:11434",
        redis_client=rdb
    )
    
    # Fix 2: Shared connection pool with high limits for fan-out
    # Mirrors Go's http.DefaultClient which reuses connections across goroutines
    http_pool = httpx.AsyncClient(
        timeout=httpx.Timeout(SHARD_TIMEOUT, connect=2.0),
        limits=httpx.Limits(max_connections=200, max_keepalive_connections=50),
    )

    # Initialize Routing Mapper (Phase 4)
    if etcd3:
        hosts = etcd_host_global.split(",")
        log.info(f"Initializing HotTermMapper with etcd hosts: {hosts}")
        try:
            client = etcd3.client(host=hosts[0].strip(), port=2379)
            global routing_mapper
            routing_mapper = HotTermMapper(client, redis_client=rdb)
            log.info("HotTermMapper created, performing initial refresh...")
            routing_mapper.refresh()
            log.info(f"Initial refresh complete. Terms: {list(routing_mapper.hot_terms.keys())}")
            asyncio.create_task(hot_term_refresh_task())
        except Exception as e:
            log.error(f"Failed to initialize HotTermMapper: {e}")
    else:
        log.warning("etcd3 not installed, HotTermMapper disabled.")

@app.on_event("shutdown")
async def shutdown_event():
    if embed_client:
        await embed_client.close()
    if http_pool:
        await http_pool.aclose()

def discover_shards(client) -> Dict[int, str]:
    shards = {}
    try:
        events = client.get_prefix("/shards/active/")
        for value, metadata in events:
            key_str = metadata.key.decode("utf-8")
            try:
                shard_id = int(key_str.split("/")[-1])
                shards[shard_id] = value.decode("utf-8")
            except ValueError:
                pass
    except Exception as e:
        log.error(f"Failed to discover shards: {e}")
    return shards

async def query_shard(addr: str, q: str, limit: int) -> List[Dict[str, Any]]:
    """Query a single shard using the shared connection pool.
    
    Fix 3: Per-shard timeout with graceful degradation.
    If a shard is slow, we return [] instead of blocking the entire fan-out.
    Mirrors Go's per-goroutine error handling in fanoutQueryParallel().
    """
    try:
        resp = await http_pool.get(f"http://{addr}/search", params={"q": q, "limit": limit})
        resp.raise_for_status()
        data = resp.json()
        return data.get("hits", [])
    except httpx.TimeoutException:
        log.warning(f"Shard {addr} timed out after {SHARD_TIMEOUT}s (returning partial results)")
        return []
    except Exception as e:
        log.error(f"Error querying shard {addr}: {e}")
        return []

async def query_shard_scored(addr: str, q: str, query_vector: list, limit: int) -> List[Dict[str, Any]]:
    """Phase 3: Near-data scoring — send query + vector to shard for local scoring.
    
    POSTs to /scored_search which computes BOTH BM25 and cosine similarity
    at the shard, returning only (doc_id, bm25_score, semantic_score, title).
    
    Network savings: 800 bytes vs 153,600 bytes per shard response (192× reduction).
    """
    try:
        resp = await http_pool.post(
            f"http://{addr}/scored_search",
            json={"q": q, "query_vector": query_vector, "limit": limit},
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("hits", [])
    except httpx.TimeoutException:
        log.warning(f"Shard {addr} scored_search timed out after {SHARD_TIMEOUT}s")
        return []
    except Exception as e:
        log.error(f"Error scored_search shard {addr}: {e}")
        return []

async def do_fanout_search(q: str, limit: int) -> Dict[str, Any]:
    start = time.time()
    
    if etcd3 is None:
        return {"error": "etcd3 not installed"}
        
    shards_dict = {}
    hosts = etcd_host_global.split(",")
    client = None
    for host in hosts:
        try:
            client = etcd3.client(host=host.strip(), port=2379)
            shards_dict = discover_shards(client)
            break # Successfully discovered shards
        except Exception as e:
            log.warning(f"Failed to connect to ETCD at {host}: {e}")
            continue
            
    if not shards_dict:
        log.warning("No active shards found in etcd (or all etcd nodes down)")
        # Still return a valid response format just with 0 hits
        return {"query": q, "total_hits": 0, "hits": [], "took": "0.0ms", "shards_queried": 0}

    # Hot Term Routing logic (Phase 4)
    routing_type = "cold"
    target_shards = []
    
    if routing_mapper:
        active_ids = list(shards_dict.keys())
        target_ids, is_hot = routing_mapper.get_target_shards(q, active_ids)
        if is_hot:
            target_shards = [shards_dict[sid] for sid in target_ids]
            routing_type = "hot"
        else:
            target_shards = list(shards_dict.values())
    else:
        target_shards = list(shards_dict.values())

    # Use shared connection pool (Fix 2)
    tasks = [query_shard(addr, q, limit) for addr in target_shards]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    all_hits = []
    for addr, result in zip(target_shards, results):
        if isinstance(result, list):
            for hit in result:
                hit["shard"] = addr
                all_hits.append(hit)

    # 4. Self-Learning (Phase 4 Extension)
    if routing_type == "cold" and routing_mapper:
        asyncio.create_task(routing_mapper.record_and_maybe_promote(q, all_hits))
    # Fix 4: Hot-term stats update (mirrors Go updateHotTermStats)
    elif routing_type == "hot" and routing_mapper:
        routing_mapper.update_stats(q, len(all_hits), len(target_shards))

    # Merge Top K
    sorted_hits = sorted(all_hits, key=lambda h: h.get("score", 0), reverse=True)[:limit]

    took = time.time() - start
    log.info("Coordinator fan-out '%s' across %d shards in %.3fs (routing: %s)", q, len(target_shards), took, routing_type)

    return {
        "query": q,
        "total_hits": len(sorted_hits),
        "hits": sorted_hits,
        "took": f"{took*1000:.1f}ms",
        "shards_queried": len(target_shards),
        "routing_type": routing_type
    }

async def cached_search(q: str, limit: int):
    if rdb is None:
        # Fallback if redis is not available
        result = await do_fanout_search(q, limit)
        return result, "DISABLED"

    # Compute a 64-character SHA256 hex string for the query
    q_hash = hashlib.sha256(q.encode("utf-8")).hexdigest()
    cache_key = f"cache:{q_hash}"
    
    # Check cache
    cached = await rdb.get(cache_key)
    if cached:
        return orjson.loads(gzip.decompress(cached)), "HIT"
    
    # Thundering herd lock
    lock_key = f"{cache_key}:lock"
    # redis-py set returns True/False for NX
    got_lock = await rdb.set(lock_key, "1", nx=True, ex=2)
    
    if not got_lock:
        await asyncio.sleep(0.05)
        cached = await rdb.get(cache_key)
        if cached:
            return orjson.loads(gzip.decompress(cached)), "HIT_WAIT"
    
    # Cache miss — do fan-out
    result = await do_fanout_search(q, limit)
    
    # Compress with gzip before storing
    compressed = gzip.compress(orjson.dumps(result))
    await rdb.setex(cache_key, CACHE_TTL, compressed)
    if got_lock:
        await rdb.delete(lock_key)
    
    return result, "MISS"

async def do_hybrid_search(q: str, limit: int, alpha: float, fusion: str) -> Dict[str, Any]:
    """Phase 3: Hybrid search with near-data scoring.
    
    BEFORE Phase 3 (what we had):
      1. Coordinator embeds query
      2. Fan-out GET /search to shards → shards return (doc_id, score, title, title_vector)
      3. Coordinator computes cosine similarity on ALL returned vectors
      4. Coordinator fuses BM25 + semantic scores
      Network: 100 hits × 1,536 bytes × 8 shards = 1.2MB per query
    
    AFTER Phase 3 (now):
      1. Coordinator embeds query
      2. Fan-out POST /scored_search to shards with query_vector
      3. Shards compute BOTH BM25 + cosine locally, return (doc_id, bm25_score, semantic_score)
      4. Coordinator only does fusion on pre-computed scores
      Network: 100 hits × 8 bytes × 8 shards = 6.4KB per query (192× reduction)
    """
    start = time.time()
    
    # 1. Get query embedding (4-tier cache: memory → redis → in-flight dedup → ollama)
    query_vector = None
    query_vector_list = []
    if embed_client:
        try:
            query_vector = await embed_client.get_embedding(q)
            query_vector_list = query_vector if isinstance(query_vector, list) else list(query_vector)
        except Exception as e:
            log.warning(f"Embedding failed for '{q}': {e} (falling back to keyword-only)")
            
    # 2. Shard Discovery
    shards_dict = {}
    hosts = etcd_host_global.split(",")
    client = None
    for host in hosts:
        try:
            client = etcd3.client(host=host.strip(), port=2379)
            shards_dict = discover_shards(client)
            break
        except Exception:
            continue
            
    if not shards_dict:
        return {"query": q, "total_hits": 0, "hits": [], "took": "0.0ms", "shards_queried": 0}

    # 3. Hot Term Routing
    routing_type = "cold"
    target_shards = []
    if routing_mapper:
        active_ids = list(shards_dict.keys())
        target_ids, is_hot = routing_mapper.get_target_shards(q, active_ids)
        if is_hot:
            target_shards = [shards_dict[sid] for sid in target_ids]
            routing_type = "hot"
        else:
            target_shards = list(shards_dict.values())
    else:
        target_shards = list(shards_dict.values())

    # 4. Fan-out to /scored_search — scoring happens at shards (Phase 3)
    retrieval_limit = 100 if fusion == "rrf" else limit * 3
    tasks = [
        query_shard_scored(addr, q, query_vector_list, retrieval_limit)
        for addr in target_shards
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    all_hits = []
    shards_responded = 0
    for addr, result in zip(target_shards, results):
        if isinstance(result, list):
            shards_responded += 1
            for hit in result:
                hit["shard"] = addr
                all_hits.append(hit)

    # 5. Fusion — scores are already computed, coordinator just merges
    if fusion == "rrf" and query_vector:
        hybrid_hits = fuse_with_rrf(all_hits, query_vector, limit)
    else:
        hybrid_hits = fuse_with_weights(all_hits, query_vector, alpha, limit)

    # 6. Self-Learning (Phase 4 Extension)
    if routing_type == "cold" and routing_mapper:
        asyncio.create_task(routing_mapper.record_and_maybe_promote(q, all_hits))
    elif routing_type == "hot" and routing_mapper:
        routing_mapper.update_stats(q, len(all_hits), len(target_shards))

    took = time.time() - start
    log.info("Coordinator HYBRID [%s] '%s' across %d/%d shards in %.3fs (routing: %s, scoring: near-data)",
             fusion, q, shards_responded, len(target_shards), took, routing_type)

    return {
        "query": q,
        "keyword_hits": len(all_hits),
        "fusion_method": fusion,
        "fusion_alpha": alpha if fusion != "rrf" else None,
        "hits": hybrid_hits,
        "took": f"{took*1000:.1f}ms",
        "shards_queried": len(target_shards),
        "shards_responded": shards_responded,
        "routing_type": routing_type,
        "scoring": "near-data",  # Phase 3 indicator
    }

@app.get("/search")
async def search_handler(
    q: str = Query(None, description="Search query"),
    limit: int = Query(20, ge=1, le=1000, description="Number of results"),
    response: Request = None,
):
    if not q:
        return JSONResponse(
            status_code=400,
            content={"error": "missing 'q' parameter"},
        )
    
    result, cache_state = await cached_search(q, limit)
    # Note: Adding custom header to the response could be done with FastAPI Response,
    # but we will just add it to the JSON body for easier verification based on the spec
    result["cache"] = cache_state
    return result

@app.get("/hybrid")
async def hybrid_handler(
    q: str = Query(None, description="Search query"),
    limit: int = Query(20, ge=1, le=1000, description="Number of results"),
    alpha: float = Query(fusion_alpha_default, ge=0.0, le=1.0, description="Weight for BM25 score (1-alpha for semantic)"),
    fusion: str = Query("rrf", regex="^(rrf|weighted)$", description="Fusion method"),
):
    if not q:
        return JSONResponse(status_code=400, content={"error": "missing 'q' parameter"})
    
    # Fix 5: Hybrid search caching via Redis
    if rdb:
        q_hash = hashlib.sha256(f"{q}:{limit}:{fusion}".encode("utf-8")).hexdigest()
        cache_key = f"hybrid:{q_hash}"
        try:
            cached = await rdb.get(cache_key)
            if cached:
                result = orjson.loads(gzip.decompress(cached))
                result["cache"] = "HIT"
                return result
        except Exception:
            pass

    result = await do_hybrid_search(q, limit, alpha, fusion)
    
    # Store in cache
    if rdb:
        try:
            compressed = gzip.compress(orjson.dumps(result))
            await rdb.setex(f"hybrid:{hashlib.sha256(f'{q}:{limit}:{fusion}'.encode('utf-8')).hexdigest()}",
                           HYBRID_CACHE_TTL, compressed)
        except Exception:
            pass
    
    result["cache"] = "MISS"
    return result

@app.get("/health")
async def health():
    return "OK"

@app.get("/routing")
async def routing_status():
    return {
        "hot_terms": routing_mapper.hot_terms if routing_mapper else {},
        "last_refresh": routing_mapper.last_refresh if routing_mapper else 0,
        "enabled": routing_mapper is not None
    }

@app.get("/cluster_stats")
async def cluster_stats():
    """Aggregate statistics from all active shards."""
    # Discover shards
    shards = []
    hosts = etcd_host_global.split(",")
    for host in hosts:
        try:
            client = etcd3.client(host=host.strip(), port=2379)
            shards_dict = discover_shards(client)
            shards = list(shards_dict.values())
            break
        except Exception:
            continue
            
    async def get_shard_stats(addr):
        try:
            resp = await http_pool.get(f"http://{addr}/stats")
            return resp.json()
        except Exception as e:
            return {"addr": addr, "status": f"down: {str(e)}"}

    results = await asyncio.gather(*[get_shard_stats(addr) for addr in shards])
    total_docs = sum(r.get("doc_count", 0) for r in results if isinstance(r, dict))
    
    return {
        "total_shards": len(shards),
        "total_documents": total_docs,
        "shards": results,
        "hot_terms_count": len(routing_mapper.hot_terms) if routing_mapper else 0
    }

@app.get("/shards")
async def shards_handler():
    if etcd3 is None:
        return {"count": 0, "active_shards": []}
        
    active = []
    hosts = etcd_host_global.split(",")
    for host in hosts:
        try:
            client = etcd3.client(host=host.strip(), port=2379)
            active = discover_shards(client)
            break
        except Exception:
            continue
            
    return {
        "count": len(active),
        "active_shards": list(active.values()) if isinstance(active, dict) else active
    }

def main():
    uvloop.install()
    global etcd_host_global, redis_host_global
    parser = argparse.ArgumentParser(description="Coordinator search server")
    parser.add_argument("--port", type=int, default=8090, help="HTTP port")
    parser.add_argument("--etcd-host", default="localhost", help="etcd server hostname")
    parser.add_argument("--redis-host", default="localhost", help="redis server hostname")
    args = parser.parse_args()

    etcd_host_global = args.etcd_host
    redis_host_global = args.redis_host

    log.info("Coordinator service ready :%d", args.port)

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=args.port,
        log_level="info",
    )

if __name__ == "__main__":
    main()
