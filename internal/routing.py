import logging
from typing import List, Dict, Tuple, Optional
import time

log = logging.getLogger(__name__)

class HotTermMapper:
    """
    Manages Hot-Term Shard Affinity.
    Maps high-frequency search terms to specific shards where data is likely located,
    reducing fan-out to all shards.
    """
    def __init__(self, etcd_client, redis_client=None):
        self.etcd_client = etcd_client
        self.rdb = redis_client
        self.hot_terms: Dict[str, List[int]] = {}
        self.last_refresh = 0
        self.refresh_interval = 30  # seconds
        self.promotion_threshold = 5  # Promote after 5 hits


    def refresh(self):
        """Fetch all hot terms from etcd prefix /hot_terms/."""
        if self.etcd_client is None:
            return

        try:
            log.info("Refreshing hot terms from etcd prefix /hot_terms/...")
            new_hot_terms = {}
            # Get all keys under /hot_terms/
            events = self.etcd_client.get_prefix("/hot_terms/")
            for value, metadata in events:
                key_str = metadata.key.decode("utf-8")
                log.info(f"Found etcd key: {key_str}")
                # Expected key format: /hot_terms/{term}/shards
                if key_str.endswith("/shards"):
                    term = key_str.split("/")[-2]
                    try:
                        shard_ids = [int(s.strip()) for s in value.decode("utf-8").split(",")]
                        new_hot_terms[term] = shard_ids
                        log.info(f"Loaded hot term: '{term}' -> {shard_ids}")
                    except ValueError:
                        continue
            
            self.hot_terms = new_hot_terms
            self.last_refresh = time.time()
            log.info("Refreshed %d hot terms from etcd.", len(self.hot_terms))
        except Exception as e:
            log.error(f"Failed to refresh hot terms: {e}")

    def get_target_shards(self, q: str, active_shard_ids: List[int]) -> Tuple[List[int], bool]:
        """
        Determines which shards to query.
        Returns (shard_ids, is_hot).
        """
        # Simple exact match for now (mirroring Go)
        term = q.lower().strip()
        if term in self.hot_terms:
            target_ids = self.hot_terms[term]
            # Intersect with currently active shards
            valid_ids = [sid for sid in target_ids if sid in active_shard_ids]
            if valid_ids:
                return valid_ids, True
        
        return active_shard_ids, False

    def update_stats(self, q: str, hit_count: int, shard_count: int):
        """Optional: Update term usage stats in etcd (as seen in original Go)."""
        if self.etcd_client is None:
            return
            
        term = q.lower().strip()
        key = f"/hot_terms/{term}/stats"
        value = f"hits:{hit_count} shards:{shard_count} ts:{int(time.time())}"
        try:
            self.etcd_client.put(key, value)
        except Exception:
            pass

    async def record_and_maybe_promote(self, q: str, shard_hits: List[Dict]):
        """
        Self-Learning: Tracks query frequency and shard distribution.
        If a term consistently hits the same shards, promotes it to 'hot'.
        """
        if not self.rdb or not self.etcd_client:
            return

        term = q.lower().strip()
        # Only analyze terms with actual results
        if not shard_hits:
            return

        # Identify which shards returned results
        # Assuming hit has 'shard_id' or we parse it from 'shard' addr
        shards_with_data = set()
        for hit in shard_hits:
            shard_addr = hit.get("shard", "")
            if "shard-" in shard_addr:
                try:
                    sid = int(shard_addr.split("-")[1].split(":")[0])
                    shards_with_data.add(sid)
                except (ValueError, IndexError):
                    continue

        if not shards_with_data:
            return

        # Increment frequency in Redis
        stat_key = f"stats:term:{term}"
        count = await self.rdb.hincrby(stat_key, "count", 1)
        
        # Store the shard set (comma separated string in a hash field)
        existing_shards_str = await self.rdb.hget(stat_key, "shards")
        current_shards = set()
        if existing_shards_str:
            current_shards = set(int(s) for s in existing_shards_str.decode().split(",") if s)
        
        current_shards.update(shards_with_data)
        new_shards_str = ",".join(map(str, sorted(list(current_shards))))
        await self.rdb.hset(stat_key, "shards", new_shards_str)

        # Promotion Logic
        if count >= self.promotion_threshold:
            # If the term is concentrated in a subset of shards (e.g. < 50% of total)
            # Or if it's just very frequent, we lock it to these shards.
            # In our case, if it's not in all shards, it's worth promoting.
            # (Assuming 8 shards total)
            if len(current_shards) < 8:
                log.info(f"AUTO-PROMOTING '{term}' to Hot Term (shards: {new_shards_str}, count: {count})")
                etcd_key = f"/hot_terms/{term}/shards"
                try:
                    # Use a 24-hour lease for adaptive re-learning
                    lease = self.etcd_client.lease(60 * 60 * 24)
                    self.etcd_client.put(etcd_key, new_shards_str, lease=lease)
                    # Also update our local cache immediately for this term
                    self.hot_terms[term] = list(current_shards)
                    log.info(f"Promoted '{term}' with 24h lease for adaptive routing.")
                except Exception as e:
                    log.error(f"Failed to auto-promote term '{term}': {e}")

