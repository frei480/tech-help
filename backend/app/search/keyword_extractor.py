"""Extract key terms from retrieved document chunks for query refinement.

This module provides LLM-based extraction of technical keywords from
document fragments. The extracted keywords are used to enrich subsequent
lexical search passes, creating a feedback loop between semantic and
keyword-based retrieval.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache

from app.llm.client import get_chat_model

logger = logging.getLogger(__name__)

EXTRACTION_PROMPT = """\
You are analyzing technical documentation fragments about T-FLEX DOCs (a PLM/PDM system).
Extract up to 10 key technical terms, feature names, UI element names, or concepts
that appear in the following text fragments.

Rules:
- Return ONLY a valid JSON array of strings, no other text or markdown fences
- Each term should be 1-4 words
- Prefer specific technical terms over generic words
- Include Russian terms as they appear in the text
- If no meaningful terms found, return an empty array

Text fragments:
{fragments}
"""


def extract_keywords(chunks: list[dict], max_chars: int = 3000) -> list[str]:
    """Extract key terms from a list of document chunks.

    Parameters
    ----------
    chunks:
        List of dicts, each containing at least a ``"text"`` key.
    max_chars:
        Maximum characters of concatenated text to send to the LLM.

    Returns
    -------
    list[str]:
        Extracted keyword phrases (deduplicated, lowercased).
    """
    if not chunks:
        return []

    # Concatenate text from chunks, truncate to max_chars
    text = "\n\n".join(c.get("text", "") for c in chunks)[:max_chars]

    try:
        prompt = EXTRACTION_PROMPT.format(fragments=text)
        result = get_chat_model(temperature=0.1).invoke(
            [{"role": "user", "content": prompt}]
        )
        raw = str(result.content).strip()

        start = raw.find("[")
        end = raw.rfind("]")
        if start != -1 and end != -1:
            keywords: list[str] = json.loads(raw[start : end + 1])
            # Normalise and deduplicate
            seen: set[str] = set()
            deduped: list[str] = []
            for kw in keywords:
                normalised = kw.strip().lower()
                if normalised and len(normalised) > 2 and normalised not in seen:
                    seen.add(normalised)
                    deduped.append(kw.strip())
            logger.info(
                "extract_keywords: extracted %d terms: %s",
                len(deduped),
                deduped,
            )
            return deduped

        logger.warning("extract_keywords: no JSON array found in LLM output: %s", raw)
    except Exception:
        logger.exception("extract_keywords failed")

    return []


@lru_cache(maxsize=128)
def extract_keywords_cached(chunks_json: str) -> list[str]:
    """Cached version — accepts JSON-serialised chunks for hashability.

    Use this when calling from hot paths to avoid repeated LLM calls for
    the same set of chunks.
    """
    chunks = json.loads(chunks_json)
    return extract_keywords(chunks)
