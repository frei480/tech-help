from __future__ import annotations

from typing import Any


def reciprocal_rank_fusion(
    rankings: list[list[dict[str, Any]]],
    top_k: int = 10,
    rank_constant: int = 60,
) -> list[dict[str, Any]]:
    """Fuse ranked retrieval lists by reciprocal rank, deduplicating by file."""
    scores: dict[str, float] = {}
    items: dict[str, dict[str, Any]] = {}

    for ranking in rankings:
        seen_in_ranking: set[str] = set()
        for rank, item in enumerate(ranking, start=1):
            metadata = item.get("metadata", {})
            rel_path = metadata.get("rel_path")
            if not isinstance(rel_path, str) or not rel_path:
                continue
            key = rel_path.replace("\\", "/").casefold()
            if key in seen_in_ranking:
                continue
            seen_in_ranking.add(key)
            scores[key] = scores.get(key, 0.0) + 1 / (rank_constant + rank)
            items.setdefault(key, item)

    ranked = sorted(scores, key=scores.__getitem__, reverse=True)
    return [
        {**items[key], "score": scores[key]}
        for key in ranked[:top_k]
    ]


def weighted_fusion(
    lexical_hits: list[dict[str, Any]],
    vector_hits: list[dict[str, Any]],
    lexical_weight: float = 0.45,
    vector_weight: float = 0.45,
    metadata_weight: float = 0.10,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """Fuse lexical and vector search results using weighted scoring.

    Lexical hits are deduplicated by ``rel_path:line_number``.
    Vector hits are deduplicated by ``rel_path:chunk_index``.
    When a lexical hit and a vector hit point to the same chunk (same
    ``rel_path`` and ``chunk_index``), their scores are combined.
    """
    scores: dict[str, dict[str, Any]] = {}

    def add(key: str, item: dict[str, Any], score: float, source: str) -> None:
        if key not in scores:
            scores[key] = {"item": item, "scores": {}}
        scores[key]["scores"][source] = score

    # ── Lexical hits: keyed by rel_path:line_number ──────────────────────
    for hit in lexical_hits:
        meta = hit.get("metadata", {})
        key = f"{meta.get('rel_path', '')}:L{meta.get('line_number', 0)}"
        lexical_score = hit.get("lexical_score", 0.0)
        metadata_score = _metadata_boost(hit)
        add(key, hit, lexical_score * lexical_weight, "lexical")
        add(key, hit, metadata_score * metadata_weight, "metadata")

    # ── Vector hits: keyed by rel_path:chunk_index ───────────────────────
    for hit in vector_hits:
        meta = hit.get("metadata", {})
        key = f"{meta.get('rel_path', '')}:C{meta.get('chunk_index', 0)}"
        vector_score = hit.get("vector_score", 0.0)
        metadata_score = _metadata_boost(hit)
        add(key, hit, vector_score * vector_weight, "vector")
        add(key, hit, metadata_score * metadata_weight, "metadata")

    ranked = []
    for entry in scores.values():
        total = sum(entry["scores"].values())
        ranked.append({"item": entry["item"], "score": total})

    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked[:top_k]


def _metadata_boost(hit: dict[str, Any]) -> float:
    if hit.get("metadata", {}).get("heading"):
        return 1.0
    return 0.5
