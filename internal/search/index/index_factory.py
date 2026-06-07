"""
Index Factory — reads system_config.yaml and returns the right vector index.

Usage:
    from internal.search.index.index_factory import create_vector_index
    vec_index = create_vector_index("system_config.yaml")
    vec_index.build(vectors, doc_ids)

Supports:
  - quantization.enabled=false    → BruteForceIndex (numpy dot-product scan)
  - quantization.type="ivfpq"    → IVFPQIndex (FAISS 3-stage cascade)
  - index.type="diskann"          → DiskANNIndex (Vamana graph on SSD)
"""

import logging
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None

from .base_index import BaseVectorIndex

log = logging.getLogger(__name__)


def create_vector_index(
    config_path: str = "system_config.yaml",
    shard_id: int = -1,
) -> BaseVectorIndex:
    """Read system_config.yaml and return the appropriate vector index.

    Decision tree:
      1. index.type == "diskann" → DiskANNIndex (Phase 2)
      2. quantization.enabled == false → BruteForceIndex (fallback)
      3. quantization.type == "ivfpq" → IVFPQIndex (Phase 1)

    Args:
        config_path: path to system_config.yaml
        shard_id: shard ID for per-shard data directories (DiskANN needs this)
    """
    config = _load_config(config_path)

    index_cfg = config.get("index", {})
    quant_cfg = config.get("quantization", {})
    storage_cfg = config.get("storage", {})
    dimension = index_cfg.get("dimension", 384)
    index_type = index_cfg.get("type", "ivfpq")

    # ── DiskANN (Phase 2) ───────────────────────────────────────────
    if index_type == "diskann":
        from .diskann_index import DiskANNIndex

        storage_path = storage_cfg.get("path", "/data/shards")
        data_dir = f"{storage_path}/shard_{shard_id}/diskann" if shard_id >= 0 else f"{storage_path}/diskann"

        idx = DiskANNIndex(
            dimension=dimension,
            data_dir=data_dir,
            R=index_cfg.get("R", 64),
            L_build=index_cfg.get("L_build", 100),
            B=index_cfg.get("B", 0.003),
            M_pq=index_cfg.get("M_pq", 64),
            cache_ram_gb=index_cfg.get("cache_ram_gb", 4.0),
            beam_width=index_cfg.get("beam_width", 8),
            search_L=index_cfg.get("search_L", 100),
        )
        log.info("Created DiskANN vector index (dim=%d, R=%d, data_dir=%s)",
                 dimension, idx.R, data_dir)
        return idx

    # ── Quantization disabled → brute force ─────────────────────────
    if not quant_cfg.get("enabled", False):
        from .brute_force_index import BruteForceIndex

        log.info("Quantization disabled — using BruteForce vector index (dim=%d)", dimension)
        return BruteForceIndex(dimension=dimension)

    # ── IVF-PQ (Phase 1) ────────────────────────────────────────────
    quant_type = quant_cfg.get("type", "ivfpq")

    if quant_type == "ivfpq":
        from .ivfpq_index import IVFPQIndex

        idx = IVFPQIndex(
            dimension=dimension,
            nlist=quant_cfg.get("nlist", 4096),
            M=quant_cfg.get("M", 48),
            nbits=quant_cfg.get("nbits", 8),
            nprobe=quant_cfg.get("nprobe", 64),
            rerank_k=quant_cfg.get("rerank_k", 500),
        )
        log.info("Created IVF-PQ vector index (dim=%d, nlist=%d, M=%d, nprobe=%d)",
                 dimension, idx.nlist, idx.M, idx.nprobe)
        return idx

    raise ValueError(f"Unknown quantization type: {quant_type}")


def _load_config(config_path: str) -> dict:
    """Load YAML config. Falls back to empty dict if file or PyYAML missing."""
    p = Path(config_path)
    if not p.exists():
        log.warning("Config %s not found — using defaults", config_path)
        return {}

    if yaml is None:
        log.warning("PyYAML not installed — using defaults")
        return {}

    with open(p) as f:
        return yaml.safe_load(f) or {}
