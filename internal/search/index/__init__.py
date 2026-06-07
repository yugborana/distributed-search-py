"""
internal.search.index — Vector index subsystem.

Exports:
    BaseVectorIndex     — abstract base class
    create_vector_index — factory function (reads system_config.yaml)
    IVFPQIndex          — FAISS IVF-PQ 3-stage cascade (Phase 1)
    BruteForceIndex     — numpy brute-force fallback
    DiskANNIndex        — Vamana graph on SSD (Phase 2)
"""

from .base_index import BaseVectorIndex
from .index_factory import create_vector_index

__all__ = ["BaseVectorIndex", "create_vector_index"]
