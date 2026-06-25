"""
Embedding Client — supports local ONNX (fastembed) and remote Ollama.

Local mode:  ~2-5ms per query, no network hop, ~150MB dependency (onnxruntime)
Ollama mode: ~100-300ms per query, requires running Ollama container

Configure via EMBED_PROVIDER env var:
  EMBED_PROVIDER=local   → fastembed (default, recommended)
  EMBED_PROVIDER=ollama  → Ollama HTTP API
"""

import hashlib
import logging
import asyncio
import concurrent.futures
from collections import OrderedDict
from typing import List, Protocol

log = logging.getLogger(__name__)


class EmbedClient(Protocol):
    """Interface for embedding clients."""
    async def get_embedding(self, text: str) -> List[float]: ...
    async def get_embeddings(self, texts: List[str]) -> List[List[float]]: ...
    async def close(self): ...


# ── Local Embedder (fastembed + ONNX) ───────────────────────────────────────

class LocalEmbedder:
    """
    Local embedding using fastembed (ONNX runtime).
    
    ~2-5ms per query on CPU. No network hop.
    Model is downloaded on first use (~90MB) and cached to disk.
    
    PERF-06 optimizations:
      - LRU eviction (OrderedDict) instead of FIFO — keeps hot queries cached
      - Redis shared cache — both coordinator workers share embeddings
      - Dedicated ONNX thread pool — prevents inference from starving async I/O
    """

    CACHE_TTL = 3600  # 1 hour in Redis

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 redis_client=None):
        try:
            from fastembed import TextEmbedding
        except ImportError:
            raise ImportError(
                "fastembed is required for local embeddings. "
                "Install with: pip install fastembed"
            )
        
        log.info("Loading local embedding model: %s (ONNX runtime)...", model_name)
        self._model = TextEmbedding(model_name, threads=1)
        # Warm up — first inference is slower due to ONNX session init
        list(self._model.embed(["warmup"]))
        log.info("Local embedder ready.")
        
        # PERF-06: Redis as shared Tier 2 cache across coordinator workers.
        self.rdb = redis_client
        
        # PERF-06: LRU cache (OrderedDict) instead of plain dict (FIFO).
        self._cache: OrderedDict[str, List[float]] = OrderedDict()
        self._cache_max = 10_000
        
        # PERF-06: Dedicated thread pool for ONNX inference.
        self._onnx_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="onnx"
        )

        # PERF-20: In-flight embedding deduplication.
        # Under concurrent load, 50 requests for "climate" all arrive before
        # the first ONNX inference completes. Without this, all 50 submit
        # separate ONNX jobs (2-worker pool → 48 queued). With this, only 1
        # inference runs; the other 49 await the same Future.
        self._inflight: dict[str, asyncio.Future] = {}
    
    def _cache_key(self, text: str) -> str:
        return text.strip().lower()
    
    async def get_embedding(self, text: str) -> List[float]:
        """Compute embedding locally. Multi-tier cache: memory → Redis → ONNX."""
        cache_key = self._cache_key(text)
        
        # Tier 1: Local in-memory LRU cache (sub-µs)
        if cache_key in self._cache:
            self._cache.move_to_end(cache_key)  # LRU touch
            return self._cache[cache_key]
        
        # Tier 2: Redis shared cache (~0.5ms, shared across workers)
        if self.rdb:
            try:
                import orjson
                cached = await self.rdb.get(f"emb:{cache_key}")
                if cached:
                    vector = orjson.loads(cached)
                    self._put_local(cache_key, vector)
                    return vector
            except Exception:
                pass

        # PERF-20: In-flight deduplication — coalesce concurrent requests
        if cache_key in self._inflight:
            return await asyncio.shield(self._inflight[cache_key])

        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self._inflight[cache_key] = future

        try:
            # Tier 3: ONNX inference in dedicated thread pool (~2-5ms)
            result = await loop.run_in_executor(self._onnx_pool, self._embed_sync, text)

            # Store in local LRU cache
            self._put_local(cache_key, result)

            # Store in Redis for cross-worker sharing (fire-and-forget)
            if self.rdb:
                try:
                    import orjson
                    asyncio.create_task(
                        self.rdb.setex(f"emb:{cache_key}", self.CACHE_TTL, orjson.dumps(result))
                    )
                except Exception:
                    pass

            future.set_result(result)
            return result
        except Exception as e:
            future.set_exception(e)
            raise
        finally:
            self._inflight.pop(cache_key, None)
    
    def _embed_sync(self, text: str) -> List[float]:
        """Synchronous embedding — called from ONNX thread pool."""
        embeddings = list(self._model.embed([text]))
        return embeddings[0].tolist()
    
    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Batch embedding. Bypasses deduplication/cache for maximum bulk throughput."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._onnx_pool, self._embed_batch_sync, texts)

    def _embed_batch_sync(self, texts: List[str]) -> List[List[float]]:
        embeddings = list(self._model.embed(texts))
        return [e.tolist() for e in embeddings]
    
    def _put_local(self, key: str, vector: List[float]):
        """Insert into LRU cache with eviction."""
        if key in self._cache:
            self._cache.move_to_end(key)
            return
        if len(self._cache) >= self._cache_max:
            self._cache.popitem(last=False)  # evict LRU (oldest)
        self._cache[key] = vector
    
    async def close(self):
        self._onnx_pool.shutdown(wait=False)


# ── Ollama Client (HTTP API) ───────────────────────────────────────────────

class OllamaClient:
    """
    Remote embedding via Ollama HTTP API.
    ~100-300ms per cold query. Requires running Ollama container.
    """

    CACHE_TTL = 3600  # 1 hour

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "all-minilm",
                 redis_client=None):
        import aiohttp
        self.base_url = base_url
        self.model = model
        self._aiohttp = aiohttp
        self.client: aiohttp.ClientSession | None = None
        self.rdb = redis_client
        self._local_cache: dict[str, List[float]] = {}
        self._local_cache_max = 10_000
        self._inflight: dict[str, asyncio.Future] = {}

    def _cache_key(self, text: str) -> str:
        return f"embed:{hashlib.sha256(text.encode('utf-8')).hexdigest()}"

    async def get_embedding(self, text: str) -> List[float]:
        cache_key = self._cache_key(text)

        # Tier 1: Local in-memory cache
        if cache_key in self._local_cache:
            return self._local_cache[cache_key]

        # Tier 2: Redis cache
        if self.rdb:
            try:
                cached = await self.rdb.get(cache_key)
                if cached:
                    import orjson
                    vector = orjson.loads(cached)
                    self._put_local(cache_key, vector)
                    return vector
            except Exception:
                pass

        # Tier 3: In-flight deduplication
        if cache_key in self._inflight:
            return await asyncio.shield(self._inflight[cache_key])

        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self._inflight[cache_key] = future

        try:
            vector = await self._call_ollama(text)
            self._put_local(cache_key, vector)

            if self.rdb:
                try:
                    import orjson
                    await self.rdb.setex(cache_key, self.CACHE_TTL, orjson.dumps(vector))
                except Exception:
                    pass

            future.set_result(vector)
            return vector
        except Exception as e:
            future.set_exception(e)
            raise
        finally:
            self._inflight.pop(cache_key, None)

    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Fallback to concurrent single requests for Ollama API."""
        return await asyncio.gather(*(self.get_embedding(t) for t in texts))

    async def _get_client(self) -> 'aiohttp.ClientSession':
        if self.client is None:
            self.client = self._aiohttp.ClientSession(
                timeout=self._aiohttp.ClientTimeout(total=30.0)
            )
        return self.client

    async def _call_ollama(self, text: str) -> List[float]:
        import orjson
        session = await self._get_client()
        async with session.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.model, "prompt": text}
        ) as resp:
            resp.raise_for_status()
            raw = await resp.read()
            data = orjson.loads(raw)
            embedding = data.get("embedding")
            if not embedding:
                raise ValueError("Ollama returned empty embedding")
            return embedding

    def _put_local(self, key: str, vector: List[float]):
        if len(self._local_cache) >= self._local_cache_max:
            oldest = next(iter(self._local_cache))
            del self._local_cache[oldest]
        self._local_cache[key] = vector

    async def close(self):
        if self.client:
            await self.client.close()


# ── Factory ─────────────────────────────────────────────────────────────────

def create_embed_client(
    provider: str = "local",
    ollama_url: str = "http://localhost:11434",
    redis_client=None,
) -> EmbedClient:
    """Create the appropriate embedding client.
    
    provider="local"  → fastembed (ONNX, ~2-5ms, no network)
    provider="ollama" → Ollama HTTP API (~100-300ms, needs container)
    """
    if provider == "local":
        # PERF-06: Pass redis_client so LocalEmbedder can share embeddings
        # across coordinator workers via Redis (cross-process shared cache)
        return LocalEmbedder(redis_client=redis_client)
    elif provider == "ollama":
        return OllamaClient(base_url=ollama_url, redis_client=redis_client)
    else:
        raise ValueError(f"Unknown embed provider: {provider}")
