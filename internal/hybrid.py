"""
Hybrid Fusion — Phase 3: Near-Data Scoring

Fuses BM25 and semantic scores that have ALREADY been computed at the shard.
Phase 3 moves all distance computation to the shard (near-data scoring),
so these functions no longer receive or process raw vectors.

Two fusion algorithms:
  1. Weighted Sum:  score = alpha × bm25_score + (1-alpha) × semantic_score
  2. RRF:           score = 1/(k + rank_bm25) + 1/(k + rank_semantic)
"""

from typing import List, Dict, Any


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Calculate cosine similarity between two vectors using numpy.
    
    DEPRECATED in Phase 3: scoring now happens at the shard.
    Kept for backward compatibility with any code that still needs it.
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    
    import numpy as np
    vec_a = np.array(a)
    vec_b = np.array(b)
    
    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return float(dot_product / (norm_a * norm_b))


def fuse_with_weights(
    hits: List[Dict[str, Any]],
    query_vector: List[float],
    alpha: float,
    limit: int,
) -> List[Dict[str, Any]]:
    """Fuse BM25 and semantic scores using a weighted sum.
    
    Formula: hybrid_score = alpha × bm25_score + (1 - alpha) × semantic_score
    
    Phase 3: If hits already contain 'bm25_score' and 'semantic_score'
    (from shard-side scored_search), use those directly.
    Falls back to computing cosine from 'title_vector' for backward compat.
    """
    hybrid_results = []

    for hit in hits:
        # Phase 3: use pre-computed scores from shard
        if "bm25_score" in hit and "semantic_score" in hit:
            keyword_score = hit["bm25_score"]
            semantic_score = hit["semantic_score"]
        else:
            # Backward compat: old-style hits with 'score' and 'title_vector'
            keyword_score = hit.get("score", 0.0)
            semantic_score = 0.0
            title_vector = hit.get("title_vector")
            if query_vector and title_vector and len(title_vector) > 0:
                semantic_score = cosine_similarity(query_vector, title_vector)

        hybrid_score = (alpha * keyword_score) + ((1 - alpha) * semantic_score)

        result = {
            "id": hit["id"],
            "title": hit.get("title", ""),
            "keyword_score": float(keyword_score),
            "semantic_score": float(semantic_score),
            "hybrid_score": float(hybrid_score),
            "shard": hit.get("shard", "unknown"),
            "fusion_method": "weighted",
        }
        hybrid_results.append(result)

    hybrid_results.sort(key=lambda x: x["hybrid_score"], reverse=True)
    return hybrid_results[:limit]


def fuse_with_rrf(
    hits: List[Dict[str, Any]],
    query_vector: List[float],
    limit: int,
    k: int = 60,
) -> List[Dict[str, Any]]:
    """Fuse BM25 and semantic scores using Reciprocal Rank Fusion (RRF).
    
    Formula: score = 1/(k + rank_bm25) + 1/(k + rank_semantic)
    
    Phase 3: If hits already contain 'bm25_score' and 'semantic_score'
    (from shard-side scored_search), uses those directly for ranking.
    Falls back to computing cosine from 'title_vector' for backward compat.
    """
    # Extract BM25 and semantic scores
    scored_hits = []
    for hit in hits:
        if "bm25_score" in hit and "semantic_score" in hit:
            bm25 = hit["bm25_score"]
            semantic = hit["semantic_score"]
        else:
            bm25 = hit.get("score", 0.0)
            semantic = 0.0
            title_vector = hit.get("title_vector")
            if query_vector and title_vector and len(title_vector) > 0:
                semantic = cosine_similarity(query_vector, title_vector)

        scored_hits.append({
            "hit": hit,
            "bm25": bm25,
            "semantic": semantic,
        })

    # Rank by BM25
    bm25_ranked = sorted(scored_hits, key=lambda x: x["bm25"], reverse=True)
    bm25_ranks = {id(s): i + 1 for i, s in enumerate(bm25_ranked)}

    # Rank by semantic
    semantic_ranked = sorted(scored_hits, key=lambda x: x["semantic"], reverse=True)
    semantic_ranks = {id(s): i + 1 for i, s in enumerate(semantic_ranked)}

    # Compute RRF scores
    rrf_results = []
    for s in scored_hits:
        b_rank = bm25_ranks[id(s)]
        s_rank = semantic_ranks[id(s)]
        rrf_score = (1.0 / (k + b_rank)) + (1.0 / (k + s_rank))

        hit = s["hit"]
        rrf_results.append({
            "id": hit["id"],
            "title": hit.get("title", ""),
            "keyword_score": float(s["bm25"]),
            "semantic_score": float(s["semantic"]),
            "hybrid_score": float(rrf_score),
            "shard": hit.get("shard", "unknown"),
            "fusion_method": "RRF",
        })

    rrf_results.sort(key=lambda x: x["hybrid_score"], reverse=True)
    return rrf_results[:limit]
