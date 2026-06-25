"""
Hybrid Fusion — Near-Data Scoring

Fuses BM25 and semantic scores that have ALREADY been computed at the shard.

Two fusion algorithms:
  1. Weighted Sum:  score = alpha × bm25_score + (1-alpha) × semantic_score
  2. RRF:           score = 1/(k + rank_bm25) + 1/(k + rank_semantic)
"""

from typing import List, Dict, Any


def fuse_with_weights(
    hits: List[Dict[str, Any]],
    alpha: float,
    limit: int,
) -> List[Dict[str, Any]]:
    """Fuse BM25 and semantic scores using a weighted sum.
    
    Formula: hybrid_score = alpha × bm25_score + (1 - alpha) × semantic_score
    """
    hybrid_results = []

    for hit in hits:
        keyword_score = hit.get("bm25_score", hit.get("score", 0.0))
        semantic_score = hit.get("semantic_score", 0.0)

        hybrid_score = (alpha * keyword_score) + ((1 - alpha) * semantic_score)

        hybrid_results.append({
            "id": hit["id"],
            "title": hit.get("title", ""),
            "keyword_score": float(keyword_score),
            "semantic_score": float(semantic_score),
            "hybrid_score": float(hybrid_score),
            "shard": hit.get("shard", "unknown"),
        })

    hybrid_results.sort(key=lambda x: x["hybrid_score"], reverse=True)
    return hybrid_results[:limit]


def fuse_with_rrf(
    hits: List[Dict[str, Any]],
    limit: int,
    k: int = 60,
) -> List[Dict[str, Any]]:
    """Fuse BM25 and semantic scores using Reciprocal Rank Fusion (RRF).
    
    Formula: score = 1/(k + rank_bm25) + 1/(k + rank_semantic)
    
    PERF: Sorts list in-place and annotates ranks directly to avoid
    intermediate dictionary allocations and id() hashing lookups.
    """
    if not hits:
        return []

    # 1. Sort by BM25 and assign rank
    hits.sort(key=lambda x: x.get("bm25_score", x.get("score", 0.0)), reverse=True)
    for i, hit in enumerate(hits):
        hit["_b_rank"] = i + 1

    # 2. Sort by Semantic and assign rank
    hits.sort(key=lambda x: x.get("semantic_score", 0.0), reverse=True)
    for i, hit in enumerate(hits):
        hit["_s_rank"] = i + 1

    # 3. Compute RRF scores
    rrf_results = []
    for hit in hits:
        b_rank = hit["_b_rank"]
        s_rank = hit["_s_rank"]
        rrf_score = (1.0 / (k + b_rank)) + (1.0 / (k + s_rank))

        rrf_results.append({
            "id": hit["id"],
            "title": hit.get("title", ""),
            "keyword_score": float(hit.get("bm25_score", hit.get("score", 0.0))),
            "semantic_score": float(hit.get("semantic_score", 0.0)),
            "hybrid_score": float(rrf_score),
            "shard": hit.get("shard", "unknown"),
        })

    # 4. Final sort by hybrid score
    rrf_results.sort(key=lambda x: x["hybrid_score"], reverse=True)
    return rrf_results[:limit]
