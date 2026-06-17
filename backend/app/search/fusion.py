from __future__ import annotations

from typing import Any


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
