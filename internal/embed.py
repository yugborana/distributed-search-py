import hashlib
import logging
import asyncio
import httpx
from typing import List, Optional

log = logging.getLogger(__name__)

class OllamaClient:
    """
    Embedding client with Redis-backed vector cache.
    
    Mirrors: internal/embed/client.go
    
    Key improvement over Go: adds a query-vector cache so repeated queries
    (e.g. 100 concurrent "election" requests) hit Ollama only ONCE.
    """

    CACHE_TTL = 3600  # 1 hour

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "all-minilm",
                 redis_client=None):
        self.base_url = base_url
        self.model = model
        self.client = httpx.AsyncClient(timeout=30.0)
        self.rdb = redis_client
        # In-memory LRU for ultra-fast repeated lookups within the same process
        self._local_cache: dict[str, List[float]] = {}
        self._local_cache_max = 1000
        self._inflight: dict[str, asyncio.Event] = {}
        self._inflight_results: dict[str, List[float]] = {}

    def _cache_key(self, text: str) -> str:
        return f"embed:{hashlib.sha256(text.encode('utf-8')).hexdigest()}"

    async def get_embedding(self, text: str) -> List[float]:
        """
        Fetch embedding with multi-tier caching:
          1. In-memory local cache (sub-microsecond)
          2. Redis cache (sub-millisecond)
          3. In-flight dedup (prevents thundering herd on Ollama)
          4. Ollama API call (300ms+)
        """
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
                pass  # Redis down — proceed to Ollama

        # Tier 3: In-flight deduplication
        # If another coroutine is already fetching this exact embedding,
        # wait for it instead of hitting Ollama again.
        if cache_key in self._inflight:
            await self._inflight[cache_key].wait()
            if cache_key in self._inflight_results:
                return self._inflight_results[cache_key]

        # Mark this embedding as in-flight
        event = asyncio.Event()
        self._inflight[cache_key] = event

        try:
            # Tier 4: Ollama API call
            vector = await self._call_ollama(text)

            # Store in all cache tiers
            self._put_local(cache_key, vector)
            self._inflight_results[cache_key] = vector

            if self.rdb:
                try:
                    import orjson
                    await self.rdb.setex(cache_key, self.CACHE_TTL, orjson.dumps(vector))
                except Exception:
                    pass  # Redis down — we still have the vector

            return vector
        finally:
            # Signal waiting coroutines
            event.set()
            self._inflight.pop(cache_key, None)
            # Clean up inflight results after a short delay
            asyncio.get_event_loop().call_later(1.0, lambda: self._inflight_results.pop(cache_key, None))

    async def _call_ollama(self, text: str) -> List[float]:
        """Raw Ollama API call. Mirrors: embed/client.go GetEmbedding()."""
        try:
            resp = await self.client.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model, "prompt": text}
            )
            resp.raise_for_status()
            data = resp.json()
            embedding = data.get("embedding")
            if not embedding:
                raise ValueError("Ollama returned empty embedding")
            return embedding
        except Exception as e:
            raise RuntimeError(f"Failed to get embedding from Ollama: {e}")

    def _put_local(self, key: str, vector: List[float]):
        """Add to local cache with basic size eviction."""
        if len(self._local_cache) >= self._local_cache_max:
            # Evict oldest entry (FIFO)
            oldest = next(iter(self._local_cache))
            del self._local_cache[oldest]
        self._local_cache[key] = vector

    async def get_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Fetch embeddings for a list of texts in parallel, leveraging cache."""
        tasks = [self.get_embedding(text) for text in texts]
        return await asyncio.gather(*tasks)

    async def close(self):
        await self.client.aclose()
