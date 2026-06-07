"""
Document model — the core data structure that flows through the entire system.

Phase 1: id, title, body, indexed_at
Phase 5 will add: title_vector (list[float])

Mirrors: distributed-search/internal/model/doc.go
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime


@dataclass
class Doc:
    """A single searchable document.
    
    In the original Go codebase this is the `model.Doc` struct.
    Field names match the JSON keys used in .jsonl shard files.
    """
    id: str
    title: str
    body: str
    indexed_at: str = field(default_factory=lambda: datetime.now().isoformat())
    # Phase 5: Semantic Search
    title_vector: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize to a plain dict for JSON encoding."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Doc":
        """Deserialize from a JSON-decoded dict."""
        return cls(
            id=d["id"],
            title=d["title"],
            body=d["body"],
            indexed_at=d.get("indexed_at", datetime.now().isoformat()),
            title_vector=d.get("title_vector", []),
        )
