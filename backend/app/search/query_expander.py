from __future__ import annotations

import json
import logging
from functools import lru_cache

from langchain_openai import ChatOpenAI

from app.config import settings

logger = logging.getLogger(__name__)

_expander_llm = ChatOpenAI(
    base_url=settings.openai_api_base_url,
    api_key=settings.openai_api_key,
    model=settings.chat_model,
    temperature=0.3,
    timeout=30.0,
)

QUERY_EXPANSION_PROMPT = """\
You are a technical documentation search assistant for T-FLEX DOCs, a PLM/PDM system.
Given a user query, generate up to 5 alternative search queries that would help find
relevant documentation.

Rules:
- Use technical synonyms, abbreviations, and alternative phrasings commonly used in PLM/EDMS/PDM domains
- Keep each variant concise (2–8 words)
- Return ONLY a valid JSON array of strings, no other text or markdown fences
- If the query is already a short technical term, you may return an empty array

Example:
User query: управление доступом
Output: ["разграничение прав доступа", "ACL", "права пользователей", "permissions", "настройка доступа"]

User query: {query}
"""


def expand_query(query: str) -> list[str]:
    """Generate alternative search queries using LLM.

    Always includes the original query as the first element.
    Results are cached via ``lru_cache`` to avoid repeated LLM calls.
    """
    try:
        prompt = QUERY_EXPANSION_PROMPT.format(query=query)
        result = _expander_llm.invoke([{"role": "user", "content": prompt}])
        text = str(result.content).strip()

        # Locate the JSON array in the response
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1:
            variants: list[str] = json.loads(text[start : end + 1])
            # Deduplicate against the original query (case-insensitive)
            extra = [v for v in variants if v.strip().lower() != query.strip().lower()]
            logger.info(
                "expand_query(%r) -> %d variants: %s",
                query,
                len(extra),
                extra,
            )
            return [query] + extra

        logger.warning("expand_query(%r): no JSON array found in LLM output: %s", query, text)
    except Exception:
        logger.exception("expand_query(%r) failed, falling back to original query", query)

    return [query]


@lru_cache(maxsize=128)
def expand_query_cached(query: str) -> list[str]:
    """Cached version of :func:`expand_query`."""
    return expand_query(query)
