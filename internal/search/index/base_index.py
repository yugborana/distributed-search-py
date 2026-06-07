"""
Base Vector Index — Abstract interface for all vector index implementations.

All vector indexes (IVF-PQ, DiskANN, brute-force) must implement this interface.
The index_factory reads system_config.yaml and returns the appropriate implementation.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np


class BaseVectorIndex(ABC):
    """Abstract base class for vector similarity indexes.

    Every implementation must support:
      - build():  bulk-load vectors (called once at startup or re-index)
      - add():    add a single vector (called on new document writes)
      - search(): return top-k nearest vectors by similarity
      - save()/load(): persist to and restore from disk
      - size:     number of vectors currently indexed
    """

    @abstractmethod
    def build(self, vectors: np.ndarray, doc_ids: List[str]) -> None:
        """Build the index from a batch of vectors.

        Args:
            vectors: shape (N, D), dtype float32. N documents, D dimensions.
            doc_ids: list of N document ID strings, parallel to vectors.
        """
        pass

    @abstractmethod
    def add(self, vector: np.ndarray, doc_id: str) -> None:
        """Add a single vector to a trained index.

        Args:
            vector: shape (D,) or (1, D), dtype float32.
            doc_id: the document's unique ID string.
        """
        pass

    @abstractmethod
    def search(self, query: np.ndarray, k: int, filter_bitmap=None) -> List[Tuple[str, float]]:
        """Find the k nearest vectors to the query.

        Args:
            query:  shape (D,) or (1, D), dtype float32.
            k:      number of results to return.
            filter_bitmap: reserved for Phase 6 (ACORN). Ignored for now.

        Returns:
            List of (doc_id, similarity_score) tuples, sorted descending by score.
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist the index to disk at the given directory path."""
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Load a previously saved index from disk."""
        pass

    @property
    @abstractmethod
    def size(self) -> int:
        """Number of vectors currently in the index."""
        pass
