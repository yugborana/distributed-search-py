"""
Coordinator

Discovers shards via etcd and fans out search queries to all of them,
merging the results. Includes straggler mitigation and per-stage profiling.
"""

import argparse
import asyncio
import orjson
import logging
import time
import hashlib
import gzip
import os
import sys
from collections import OrderedDict
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Dict, Any

from internal.embed import create_embed_client
from internal.hybrid import fuse_with_weights, fuse_with_rrf

from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse, ORJSONResponse
import uvicorn
import aiohttp
try:
    import grpc
    from internal.proto import shard_pb2, shard_pb2_grpc
except ImportError:
    grpc = None
    shard_pb2 = None
    shard_pb2_grpc = None

_grpc_stubs = {}
_grpc_channels = {}
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

# PERF-PYDANTIC: Use ORJSONResponse as default response class.
# FastAPI's default JSONResponse runs jsonable_encoder() + stdlib json.dumps()
# on every response. ORJSONResponse uses orjson.dumps() directly — 3-10x faster
# for numeric-heavy payloads (hit scores, vectors).
app = FastAPI(title="Distributed Search — Coordinator", default_response_class=ORJSONResponse)

# ── Prometheus Metrics ──────────────────────────────────────────────────────
try:
    from prometheus_client import Counter, Histogram, Gauge, make_asgi_app, CollectorRegistry

    search_registry = CollectorRegistry()

    REQUEST_COUNT = Counter(
        "search_requests_total", "Total search requests", ["endpoint", "status"],
        registry=search_registry
    )
    REQUEST_LATENCY = Histogram(
        "search_latency_seconds", "Search request latency", ["endpoint"],
        buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
        registry=search_registry
    )
    CACHE_HITS = Counter(
        "search_cache_hits_total", "Cache hits", ["tier"],
        registry=search_registry
    )
    CACHE_MISSES = Counter(
        "search_cache_misses_total", "Cache misses", ["tier"],
        registry=search_registry
    )
    INDEX_DOC_COUNT = Gauge(
        "search_index_doc_count", "Total indexed documents",
        registry=search_registry
    )
    EMBED_LATENCY = Histogram(
        "search_embed_latency_seconds", "Embedding generation latency",
        buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1],
        registry=search_registry
    )

    # Mount the /metrics endpoint for Prometheus scraping
    metrics_app = make_asgi_app(registry=search_registry)
    app.mount("/metrics", metrics_app)
    _METRICS_ENABLED = True
except ImportError:
    _METRICS_ENABLED = False

from fastapi import Request

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    if not _METRICS_ENABLED or request.url.path not in ["/search", "/hybrid"]:
        return await call_next(request)
        
    t_start = time.perf_counter()
    endpoint = request.url.path.strip("/")
    
    try:
        response = await call_next(request)
        status_code = str(response.status_code)
    except Exception as e:
        status_code = "500"
        REQUEST_COUNT.labels(endpoint=endpoint, status=status_code).inc()
        REQUEST_LATENCY.labels(endpoint=endpoint).observe(time.perf_counter() - t_start)
        raise e
        
    REQUEST_COUNT.labels(endpoint=endpoint, status=status_code).inc()
    REQUEST_LATENCY.labels(endpoint=endpoint).observe(time.perf_counter() - t_start)
    return response

etcd_host_global = os.environ.get("ETCD_HOSTS", "localhost")
redis_host_global = os.environ.get("REDIS_HOST", "localhost")
ollama_host_global = os.environ.get("OLLAMA_HOST", "localhost")
embed_provider = os.environ.get("EMBED_PROVIDER", "local")
fusion_alpha_default = float(os.environ.get("FUSION_ALPHA", "0.5"))

# PERF-13: In-process partition sharding eliminates all network overhead.
# Set SHARD_MODE=inprocess to load all Tantivy partitions directly.
# Set SHARD_MODE=distributed (default for backward compat) for gRPC fan-out.
shard_mode = os.environ.get("SHARD_MODE", "distributed")
index_base_path = os.environ.get("INDEX_BASE", "/app/search.idx")
num_partitions = int(os.environ.get("NUM_PARTITIONS", "8"))

rdb = None
embed_client = None
http_pool: aiohttp.ClientSession = None  # PERF-03: aiohttp replaces httpx
partition_manager = None  # PERF-13: Set during startup if SHARD_MODE=inprocess
CACHE_TTL = 300
HYBRID_CACHE_TTL = 60
SHARD_TIMEOUT = aiohttp.ClientTimeout(total=2.5, connect=1.0)
STRAGGLER_DEADLINE = 3.0  # return partial results if stragglers exceed this

# PERF-14: L0 in-memory result cache — avoids Redis RTT entirely for hot queries.
# EXTREME PERF: We cache the PRE-SERIALIZED JSON bytes. This avoids running
# orjson.dumps() 10,000 times a second for the exact same object.
_L0_CACHE_MAX = 2000
_L0_TTL = 30.0  # seconds
_l0_cache: OrderedDict[str, tuple] = OrderedDict()  # key -> (bytes, timestamp)


def _l0_get(key: str) -> bytes | None:
    """Get raw JSON bytes from L0 in-memory cache."""
    if key in _l0_cache:
        result_bytes, ts = _l0_cache[key]
        if time.monotonic() - ts < _L0_TTL:
            _l0_cache.move_to_end(key)
            return result_bytes
        else:
            del _l0_cache[key]
    return None


def _l0_put(key: str, result: dict):
    """Serialize and put into L0 in-memory cache."""
    # We pre-bake the "L0_HIT" tag into the bytes so we never have to parse it
    result_copy = dict(result)
    result_copy["cache"] = "L0_HIT"
    result_bytes = orjson.dumps(result_copy)
    
    if key in _l0_cache:
        _l0_cache.move_to_end(key)
        _l0_cache[key] = (result_bytes, time.monotonic())
        return
    if len(_l0_cache) >= _L0_CACHE_MAX:
        _l0_cache.popitem(last=False)
    _l0_cache[key] = (result_bytes, time.monotonic())


# PERF-15: In-flight request deduplication.
# Under concurrent load, 50 requests for the same query should NOT trigger
# 50 separate searches. Only 1 computes; the other 49 await the same Future.
_inflight_searches: dict[str, asyncio.Future] = {}
_inflight_hybrid: dict[str, asyncio.Future] = {}

# PERF-10: Threshold for gzip compression. Payloads smaller than this
# are stored raw — gzip overhead (~0.1-0.5ms CPU) isn't worth the ~30%
# size reduction on a 2KB payload (saves ~600 bytes in Redis).
_COMPRESS_THRESHOLD = 2048


def _cache_compress(data: dict) -> bytes:
    """Serialize + conditionally compress for Redis storage.
    
    PERF-10: Uses a 1-byte prefix to auto-detect format on read:
      b'R' + raw orjson bytes  (payloads <= 2KB)
      b'G' + gzip bytes        (payloads > 2KB)
    """
    raw = orjson.dumps(data)
    if len(raw) > _COMPRESS_THRESHOLD:
        return b'G' + gzip.compress(raw, compresslevel=1)  # level 1 = fast
    return b'R' + raw


def _cache_decompress(data: bytes) -> dict:
    """Decompress + deserialize from Redis storage."""
    if data[0:1] == b'G':
        return orjson.loads(gzip.decompress(data[1:]))
    elif data[0:1] == b'R':
        return orjson.loads(data[1:])
    else:
        # Legacy format (no prefix) — assume gzip
        return orjson.loads(gzip.decompress(data))

# ── Cached shard topology ───────────────────────────────────────────────────
_shard_cache: Dict[int, str] = {}
_shard_cache_ts: float = 0
_SHARD_CACHE_TTL = 5.0
_etcd_client = None


def _get_etcd_client():
    """Get or create a reusable etcd client."""
    global _etcd_client
    if _etcd_client is not None:
        return _etcd_client
    if etcd3 is None:
        return None
    hosts = etcd_host_global.split(",")
    for host in hosts:
        try:
            _etcd_client = etcd3.client(host=host.strip(), port=2379)
            return _etcd_client
        except Exception as e:
            log.warning(f"Failed to connect to etcd at {host}: {e}")
    return None


def _discover_shards_sync() -> Dict[int, str]:
    """Synchronous etcd shard discovery. Runs in thread pool."""
    client = _get_etcd_client()
    if client is None:
        return {}
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
        log.warning(f"etcd shard discovery failed: {e}")
        global _etcd_client
        _etcd_client = None
    return shards


async def get_shards() -> Dict[int, str]:
    """Get shard topology with caching. Refreshes every 5s."""
    global _shard_cache, _shard_cache_ts
    now = time.time()
    if now - _shard_cache_ts < _SHARD_CACHE_TTL and _shard_cache:
        return _shard_cache
    loop = asyncio.get_running_loop()
    _shard_cache = await loop.run_in_executor(None, _discover_shards_sync)
    _shard_cache_ts = now
    
    # Evict dead gRPC channels when topology changes
    active_addrs = set(_shard_cache.values())
    dead_addrs = set(_grpc_stubs.keys()) - active_addrs
    for addr in dead_addrs:
        _grpc_stubs.pop(addr, None)
        channel = _grpc_channels.pop(addr, None)
        if channel:
            try:
                await channel.close()
            except Exception:
                pass

    return _shard_cache


# ── Startup / Shutdown ──────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    global rdb, embed_client, http_pool, partition_manager

    if aioredis is not None:
        redis_url = os.environ.get("REDIS_URL", f"redis://{redis_host_global}:6379")
        # In Kubernetes envFrom, $(VAR) is not evaluated. We interpolate manually.
        if "$(REDIS_PASSWORD)" in redis_url:
            redis_url = redis_url.replace("$(REDIS_PASSWORD)", os.environ.get("REDIS_PASSWORD", ""))
        rdb = aioredis.from_url(redis_url, decode_responses=False)

    embed_client = create_embed_client(
        provider=embed_provider,
        ollama_url=f"http://{ollama_host_global}:11434",
        redis_client=rdb,
    )
    log.info("Embedding provider: %s", embed_provider)

    if shard_mode == "inprocess":
        # PERF-13: Load all Tantivy partitions in-process.
        # Zero network overhead — searches go directly to ThreadPoolExecutor.
        from internal.partition_manager import PartitionManager
        partition_manager = PartitionManager(
            index_base=index_base_path,
            num_partitions=num_partitions,
            max_workers=num_partitions,
        )
        log.info("SHARD_MODE=inprocess: %d partitions loaded, %d total docs",
                 len(partition_manager.partitions), partition_manager.total_docs())
        if _METRICS_ENABLED:
            INDEX_DOC_COUNT.set(partition_manager.total_docs())

        # Automatically refresh the in-process searchers to pick up newly indexed docs
        async def refresh_partitions_task():
            loop = asyncio.get_running_loop()
            while True:
                await asyncio.sleep(30)
                if partition_manager:
                    await loop.run_in_executor(None, partition_manager.reload_all)
                    if _METRICS_ENABLED:
                        INDEX_DOC_COUNT.set(partition_manager.total_docs())

        asyncio.create_task(refresh_partitions_task())

        if partition_manager is not None and len(partition_manager.partitions) > 0:
            async def _warmup():
                await asyncio.sleep(2)   # tiny delay for event loop to settle
                try:
                    # Prime OS page cache and ONNX session
                    await partition_manager.scored_search_all("warmup", [], limit=1)
                    log.info("Warmup complete - index pages cached, ONNX session warm")
                except Exception as e:
                    log.warning("Warmup failed (non-fatal): %s", e)
            asyncio.create_task(_warmup())
    else:
        # Distributed mode: fan-out via gRPC/HTTP to separate shard containers
        http_pool = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(
                limit=200,
                keepalive_timeout=30,
                enable_cleanup_closed=True,
            ),
            timeout=SHARD_TIMEOUT,
        )
        await get_shards()
        log.info("SHARD_MODE=distributed: Shards discovered: %d", len(_shard_cache))


@app.on_event("shutdown")
async def shutdown_event():
    if embed_client:
        await embed_client.close()
    if http_pool:
        await http_pool.close()


# ── Shard Query Helpers ─────────────────────────────────────────────────────

def _get_grpc_stub(addr: str):
    if addr not in _grpc_stubs:
        host, port_str = addr.split(":")
        grpc_addr = f"{host}:{int(port_str) + 1000}"
        channel = grpc.aio.insecure_channel(grpc_addr)
        _grpc_channels[addr] = channel
        _grpc_stubs[addr] = shard_pb2_grpc.ShardServiceStub(channel)
    return _grpc_stubs[addr]

async def query_shard(addr: str, q: str, limit: int) -> List[Dict[str, Any]]:
    """Query a single shard for BM25 results via gRPC."""
    if grpc and shard_pb2_grpc:
        try:
            stub = _get_grpc_stub(addr)
            req = shard_pb2.SearchRequest(query=q, limit=limit)
            resp = await stub.Search(req, timeout=2.0)
            return [
                {"id": h.id, "score": round(h.score, 6), "title": h.title}
                for h in resp.hits
            ]
        except Exception as e:
            log.warning(f"gRPC query_shard {addr} failed: {e}")
            return []

    try:
        async with http_pool.get(
            f"http://{addr}/search",
            params={"q": q, "limit": limit},
        ) as resp:
            if resp.status != 200:
                return []
            raw = await resp.read()
            data = orjson.loads(raw)
            return data.get("hits", [])
    except Exception as e:
        log.error(f"Error querying shard {addr}: {e}")
        return []


async def query_shard_scored(addr: str, body_bytes: bytes) -> List[Dict[str, Any]]:
    """Near-data scoring — send packed protobuf vector body to shard via gRPC."""
    if grpc and shard_pb2_grpc:
        try:
            body_dict = orjson.loads(body_bytes)
            stub = _get_grpc_stub(addr)
            req = shard_pb2.ScoredSearchRequest(
                query=body_dict["q"],
                query_vector=body_dict.get("query_vector", []),
                limit=body_dict.get("limit", 50)
            )
            resp = await stub.ScoredSearch(req, timeout=2.0)
            return [
                {
                    "id": h.id,
                    "bm25_score": round(h.bm25_score, 6),
                    "semantic_score": round(h.semantic_score, 6),
                    "score": round(h.score, 6),
                    "title": h.title
                }
                for h in resp.hits
            ]
        except Exception as e:
            log.warning(f"gRPC query_shard_scored {addr} failed: {e}")
            return []

    try:
        async with http_pool.post(
            f"http://{addr}/scored_search",
            data=body_bytes,
            headers={"Content-Type": "application/json"},
        ) as resp:
            if resp.status != 200:
                return []
            raw = await resp.read()
            data = orjson.loads(raw)
            return data.get("hits", [])
    except Exception as e:
        log.error(f"Error scored_search shard {addr}: {e}")
        return []


# ── Fan-out with straggler mitigation ───────────────────────────────────────

async def fanout_with_deadline(tasks: list, deadline: float) -> list:
    """Run tasks in parallel, return results from shards that respond within deadline.
    
    If all shards respond before the deadline, all results are used.
    If some shards are slow (stragglers), we return partial results after
    the deadline to avoid p99 latency spikes.
    """
    wrapped = [asyncio.ensure_future(t) for t in tasks]
    done, pending = await asyncio.wait(wrapped, timeout=deadline)

    results = []
    for task in done:
        try:
            results.append(task.result())
        except Exception:
            results.append([])

    # Cancel stragglers — don't waste resources
    stragglers = len(pending)
    for task in pending:
        task.cancel()
        results.append([])  # empty result for cancelled shards

    if stragglers > 0:
        log.warning(f"Straggler mitigation: {stragglers} shard(s) cancelled after {deadline}s deadline")

    return results


# ── Core Search Logic ───────────────────────────────────────────────────────

async def do_fanout_search(q: str, limit: int) -> Dict[str, Any]:
    start = time.perf_counter()

    # PERF-13: In-process partition search — zero network overhead
    if partition_manager is not None:
        sorted_hits, partitions_responded = await partition_manager.search_all(q, limit)
        took_ms = (time.perf_counter() - start) * 1000
        return {
            "query": q,
            "total_hits": len(sorted_hits),
            "hits": sorted_hits,
            "took": f"{took_ms:.1f}ms",
            "shards_queried": len(partition_manager.partitions),
            "shards_responded": partitions_responded,
        }

    # Distributed mode: fan-out via gRPC/HTTP
    shards_dict = await get_shards()
    if not shards_dict:
        return {"query": q, "total_hits": 0, "hits": [], "took": "0.0ms", "shards_queried": 0}

    target_shards = list(shards_dict.values())
    tasks = [query_shard(addr, q, limit) for addr in target_shards]
    results = await fanout_with_deadline(tasks, STRAGGLER_DEADLINE)

    all_hits = []
    shards_responded = 0
    for addr, result in zip(target_shards, results):
        if isinstance(result, list) and result:
            shards_responded += 1
            for hit in result:
                hit["shard"] = addr
                all_hits.append(hit)

    sorted_hits = sorted(all_hits, key=lambda h: h.get("score", 0), reverse=True)[:limit]

    took_ms = (time.perf_counter() - start) * 1000
    return {
        "query": q,
        "total_hits": len(sorted_hits),
        "hits": sorted_hits,
        "took": f"{took_ms:.1f}ms",
        "shards_queried": len(target_shards),
        "shards_responded": shards_responded,
    }


async def do_hybrid_search(q: str, limit: int, alpha: float, fusion: str) -> Dict[str, Any]:
    """Hybrid search with near-data scoring and per-stage profiling."""
    t0 = time.perf_counter()

    # Stage 1: Get query embedding
    query_vector = None
    query_vector_list = []
    if embed_client:
        try:
            query_vector = await embed_client.get_embedding(q)
            query_vector_list = query_vector if isinstance(query_vector, list) else list(query_vector)
        except Exception as e:
            log.warning(f"Embedding failed for '{q}': {e} (falling back to keyword-only)")
    t1 = time.perf_counter()
    if _METRICS_ENABLED:
        EMBED_LATENCY.observe(t1 - t0)

    # PERF-13: In-process partition scored search — zero network overhead
    if partition_manager is not None:
        retrieval_limit = max(limit, min(limit * 2, 200))
        all_hits, shards_responded = await partition_manager.scored_search_all(
            query=q, query_vector=query_vector_list, limit=retrieval_limit
        )
        t2 = time.perf_counter()

        actual_fusion = "rrf" if (fusion == "rrf" and query_vector) else "weighted"
        if actual_fusion == "rrf":
            hybrid_hits = fuse_with_rrf(all_hits, limit)
        else:
            hybrid_hits = fuse_with_weights(all_hits, alpha, limit)
        t3 = time.perf_counter()

        embed_ms = (t1 - t0) * 1000
        fanout_ms = (t2 - t1) * 1000
        fusion_ms = (t3 - t2) * 1000
        total_ms = (t3 - t0) * 1000

        return {
            "query": q,
            "keyword_hits": len(all_hits),
            "fusion_method": actual_fusion,
            "fusion_alpha": alpha if actual_fusion != "rrf" else None,
            "hits": hybrid_hits,
            "took": f"{total_ms:.1f}ms",
            "shards_queried": len(partition_manager.partitions),
            "shards_responded": shards_responded,
            "timing": {
                "embed_ms": round(embed_ms, 2),
                "fanout_ms": round(fanout_ms, 2),
                "fusion_ms": round(fusion_ms, 2),
                "total_ms": round(total_ms, 2),
            },
        }

    # Distributed mode: fan-out via gRPC/HTTP
    shards_dict = await get_shards()
    if not shards_dict:
        return {"query": q, "total_hits": 0, "hits": [], "took": "0.0ms", "shards_queried": 0}

    target_shards = list(shards_dict.values())

    # Stage 3: Fan-out to /scored_search with straggler mitigation
    retrieval_limit = max(limit, min(limit * 2, 200))
    body_bytes = orjson.dumps({"q": q, "query_vector": query_vector_list, "limit": retrieval_limit})
    tasks = [
        query_shard_scored(addr, body_bytes)
        for addr in target_shards
    ]
    results = await fanout_with_deadline(tasks, STRAGGLER_DEADLINE)
    t2 = time.perf_counter()

    all_hits = []
    shards_responded = 0
    for addr, result in zip(target_shards, results):
        if isinstance(result, list) and result:
            shards_responded += 1
            for hit in result:
                hit["shard"] = addr
                all_hits.append(hit)

    # Stage 4: Fusion — scores already computed at shards
    actual_fusion = "rrf" if (fusion == "rrf" and query_vector) else "weighted"
    if actual_fusion == "rrf":
        hybrid_hits = fuse_with_rrf(all_hits, limit)
    else:
        hybrid_hits = fuse_with_weights(all_hits, alpha, limit)
    t3 = time.perf_counter()

    embed_ms = (t1 - t0) * 1000
    fanout_ms = (t2 - t1) * 1000
    fusion_ms = (t3 - t2) * 1000
    total_ms = (t3 - t0) * 1000

    return {
        "query": q,
        "keyword_hits": len(all_hits),
        "fusion_method": actual_fusion,
        "fusion_alpha": alpha if actual_fusion != "rrf" else None,
        "hits": hybrid_hits,
        "took": f"{total_ms:.1f}ms",
        "shards_queried": len(target_shards),
        "shards_responded": shards_responded,
        "timing": {
            "embed_ms": round(embed_ms, 2),
            "fanout_ms": round(fanout_ms, 2),
            "fusion_ms": round(fusion_ms, 2),
            "total_ms": round(total_ms, 2),
        },
    }


# ── Caching Layer ───────────────────────────────────────────────────────────


def _fast_cache_key(prefix: str, *parts) -> str:
    """PERF-16: Fast cache key using blake2b.
    Must be deterministic across processes so Uvicorn workers share the Redis cache.
    """
    digest = hashlib.blake2b(repr(parts).encode(), digest_size=8).hexdigest()
    return f"{prefix}:{digest}"


async def _redis_set_bg(cache_key: str, ttl: int, data: dict):
    """PERF-17: Fire-and-forget Redis write. Don't block response on cache SET."""
    try:
        await rdb.setex(cache_key, ttl, _cache_compress(data))
    except Exception:
        pass


async def cached_search(q: str, limit: int):
    cache_key = _fast_cache_key("s", q, limit)

    # PERF-14: L0 in-memory cache — bypasses ORJSONResponse serialization entirely
    l0_bytes = _l0_get(cache_key)
    if l0_bytes is not None:
        from fastapi.responses import Response
        return Response(content=l0_bytes, media_type="application/json"), "L0_HIT"

    # PERF-15: In-flight deduplication — only 1 computation per unique query
    if cache_key in _inflight_searches:
        result = await _inflight_searches[cache_key]
        return dict(result), "COALESCED"

    if rdb is not None:
        try:
            cached = await rdb.get(cache_key)
            if cached:
                result = _cache_decompress(cached)
                _l0_put(cache_key, result)
                return result, "HIT"
        except Exception:
            pass

    # Create an in-flight future so concurrent requests coalesce
    loop = asyncio.get_running_loop()
    future = loop.create_future()
    _inflight_searches[cache_key] = future

    try:
        result = await do_fanout_search(q, limit)
        _l0_put(cache_key, result)
        future.set_result(result)

        # PERF-17: Fire-and-forget Redis write
        if rdb is not None:
            asyncio.create_task(_redis_set_bg(cache_key, CACHE_TTL, result))

        return result, "MISS"
    except Exception as e:
        future.set_exception(e)
        raise
    finally:
        _inflight_searches.pop(cache_key, None)


# ── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/search")
async def search_handler(
    q: str = Query(None, description="Search query"),
    limit: int = Query(20, ge=1, le=1000, description="Number of results"),
):
    if not q:
        return JSONResponse(status_code=400, content={"error": "missing 'q' parameter"})

    result_or_response, cache_state = await cached_search(q, limit)
    
    # If L0 hit, we get a pre-serialized raw Response object
    from fastapi.responses import Response
    if isinstance(result_or_response, Response):
        if _METRICS_ENABLED:
            CACHE_HITS.labels(tier="L0").inc()
        return result_or_response

    # Otherwise, it's a dict that needs serialization
    result_or_response["cache"] = cache_state
    if _METRICS_ENABLED:
        if cache_state == "HIT":
            CACHE_HITS.labels(tier="redis").inc()
        elif cache_state == "MISS":
            CACHE_MISSES.labels(tier="all").inc()
        elif cache_state == "COALESCED":
            CACHE_HITS.labels(tier="coalesced").inc()
    return result_or_response


@app.get("/hybrid")
async def hybrid_handler(
    q: str = Query(None, description="Search query"),
    limit: int = Query(20, ge=1, le=1000, description="Number of results"),
    alpha: float = Query(fusion_alpha_default, ge=0.0, le=1.0),
    fusion: str = Query("rrf", pattern="^(rrf|weighted)$"),
):
    if not q:
        return JSONResponse(status_code=400, content={"error": "missing 'q' parameter"})

    cache_key = _fast_cache_key("h", q, limit, fusion, alpha)

    # PERF-14: L0 in-memory cache — bypasses ORJSONResponse serialization entirely
    l0_bytes = _l0_get(cache_key)
    if l0_bytes is not None:
        if _METRICS_ENABLED:
            CACHE_HITS.labels(tier="L0").inc()
        from fastapi.responses import Response
        return Response(content=l0_bytes, media_type="application/json")

    # PERF-15: In-flight deduplication
    if cache_key in _inflight_hybrid:
        result = await _inflight_hybrid[cache_key]
        result = dict(result)  # shallow copy so we can mutate cache field
        result["cache"] = "COALESCED"
        if _METRICS_ENABLED:
            CACHE_HITS.labels(tier="coalesced").inc()
        return result

    if rdb is not None:
        try:
            cached = await rdb.get(cache_key)
            if cached:
                result = _cache_decompress(cached)
                _l0_put(cache_key, result)
                result["cache"] = "HIT"
                if _METRICS_ENABLED:
                    CACHE_HITS.labels(tier="redis").inc()
                return result
        except Exception:
            pass

    # Create in-flight future
    loop = asyncio.get_running_loop()
    future = loop.create_future()
    _inflight_hybrid[cache_key] = future

    try:
        result = await do_hybrid_search(q, limit, alpha, fusion)
        _l0_put(cache_key, result)
        future.set_result(result)

        # PERF-17: Fire-and-forget Redis write
        if rdb is not None:
            asyncio.create_task(_redis_set_bg(cache_key, HYBRID_CACHE_TTL, result))

        result["cache"] = "MISS"
        if _METRICS_ENABLED:
            CACHE_MISSES.labels(tier="all").inc()
        return result
    except Exception as e:
        future.set_exception(e)
        raise
    finally:
        _inflight_hybrid.pop(cache_key, None)


@app.get("/health")
async def health():
    return "OK"


@app.get("/cluster_stats")
async def cluster_stats():
    """Aggregate statistics from all active shards."""
    if partition_manager is not None:
        return {
            "mode": "inprocess",
            "total_shards": len(partition_manager.partitions),
            "total_documents": partition_manager.total_docs(),
        }

    shards_dict = await get_shards()
    shards = list(shards_dict.values())

    async def get_shard_stats(addr):
        try:
            async with http_pool.get(f"http://{addr}/stats") as resp:
                raw = await resp.read()
                return orjson.loads(raw)
        except Exception as e:
            return {"addr": addr, "status": f"down: {str(e)}"}

    results = await asyncio.gather(*[get_shard_stats(addr) for addr in shards])
    total_docs = sum(r.get("doc_count", 0) for r in results if isinstance(r, dict))

    return {
        "total_shards": len(shards),
        "total_documents": total_docs,
        "shards": results,
    }


@app.get("/shards")
async def shards_handler():
    if partition_manager is not None:
        return {
            "mode": "inprocess",
            "count": len(partition_manager.partitions),
            "active_shards": [f"local-partition-{i}" for i in range(len(partition_manager.partitions))],
        }

    active = await get_shards()
    return {
        "count": len(active),
        "active_shards": list(active.values()) if isinstance(active, dict) else active,
    }


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    global etcd_host_global, redis_host_global
    parser = argparse.ArgumentParser(description="Coordinator search server")
    parser.add_argument("--port", type=int, default=8090, help="HTTP port")
    parser.add_argument("--etcd-host", default="localhost", help="etcd server hostname")
    parser.add_argument("--redis-host", default="localhost", help="redis server hostname")
    args = parser.parse_args()

    etcd_host_global = args.etcd_host
    redis_host_global = args.redis_host

    log.info("Coordinator service ready :%d", args.port)

    # PERF-02: uvloop gives a 2-4x faster event loop on Linux (Docker).
    # On Windows, it's not available — graceful fallback.
    loop_policy = "auto"
    try:
        import uvloop  # type: ignore
        uvloop.install()  # type: ignore
        loop_policy = "uvloop"
        log.info("uvloop installed")
    except ImportError:
        pass

    # PERF-11: httptools replaces uvicorn's default h11 (pure Python) HTTP
    # parser with llhttp (C-accelerated) — ~5x faster request parsing.
    http_impl = "auto"
    try:
        import httptools  # noqa: F401
        http_impl = "httptools"
        log.info("httptools available")
    except ImportError:
        pass

    uvicorn.run(
        "cmd.coordinator:app",
        host="0.0.0.0",
        port=args.port,
        log_level="info",
        access_log=False,
        # PERF-02: Reduced from 8 to 2 workers.
        # The coordinator is I/O-bound (fan-out to shards + Redis), not CPU-bound.
        # With 8 workers, each loads a 150MB ONNX model (1.2GB total) and has its
        # own embedding cache — diluting cache hit rate by 8x.
        # With 2 workers: 300MB total, 4x better cache hit rate, and async I/O
        # handles the concurrency.
        # PERF-18: workers=2 even for inprocess mode.
        # Tantivy indexes are mmap'd files — fork() shares physical pages
        # via copy-on-write. Only the Python heap (~70MB) is duplicated,
        # NOT the ~120MB of index data. Two event loops double throughput.
        workers=2,
        loop=loop_policy,
        http=http_impl,  # PERF-11: C-accelerated HTTP parsing
    )


if __name__ == "__main__":
    main()
