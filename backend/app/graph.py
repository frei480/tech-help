import asyncio
import math
from typing import Any, TypedDict

from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from langgraph.types import StreamWriter
from openai import AsyncOpenAI


from app.config import settings
from app.llm.prompts import build_prompt
from app.search.file_loader import read_full_file
from app.search.fusion import reciprocal_rank_fusion
from app.search.keyword_extractor import extract_keywords
from app.search.query_expander import expand_query_cached
from app.search.ripgrep import run_ripgrep
from app.search.vector import VectorSearch


class GraphState(TypedDict):
    query: str
    mode: str
    model: str | None
    vector_hits: list[dict[str, Any]]
    extracted_keywords: list[str]
    bm25_hits: list[dict[str, Any]]
    lexical_hits_pass1: list[dict[str, Any]]
    lexical_keywords: list[str]
    lexical_hits_pass2: list[dict[str, Any]]
    fused: list[dict[str, Any]]
    answer: str | None


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _summarize_hits(hits: list[dict[str, Any]], max_count: int = 5) -> list[dict[str, Any]]:
    """Summarize search hits for trace display — file paths and scores only."""
    out = []
    for h in hits[:max_count]:
        meta = h.get("metadata", {})
        out.append(
            {
                "rel_path": meta.get("rel_path", "?"),
                "line_number": meta.get("line_number"),
                "score": h.get("lexical_score") or h.get("bm25_score") or h.get("score"),
                "snippet_preview": (h.get("text", "") or "")[:200],
            }
        )
    return out


def _run_lexical_search(variants: list[str]) -> list[dict[str, Any]]:
    """Run ripgrep for each query variant and rank hits by term specificity."""
    hit_scores: dict[str, dict[str, Any]] = {}

    for variant in variants:
        search_variants = _yo_variants(variant)
        variant_hits: dict[str, Any] = {}
        for search_variant in search_variants:
            for ripgrep_hit in run_ripgrep(search_variant):
                variant_hits.setdefault(
                    f"{ripgrep_hit.rel_path}:{ripgrep_hit.line_number}",
                    ripgrep_hit,
                )

        hits = list(variant_hits.values())
        document_frequency = len({ripgrep_hit.rel_path for ripgrep_hit in hits})
        if not document_frequency:
            continue
        specificity = 1 / (1 + math.log1p(document_frequency))

        for ripgrep_hit in hits:
            key = f"{ripgrep_hit.rel_path}:{ripgrep_hit.line_number}"
            if key not in hit_scores:
                hit_scores[key] = {
                    "text": ripgrep_hit.snippet,
                    "metadata": {
                        "file_path": str(ripgrep_hit.file_path),
                        "rel_path": ripgrep_hit.rel_path,
                        "line_number": ripgrep_hit.line_number,
                        "chunk_index": -1,
                    },
                    "lexical_score": 0.0,
                }
            hit_scores[key]["lexical_score"] += specificity

    if not hit_scores:
        return []

    max_score = max(hit["lexical_score"] for hit in hit_scores.values())
    ranked_hits = list(hit_scores.values())
    for hit in ranked_hits:
        hit["lexical_score"] /= max_score
    ranked_hits.sort(key=lambda hit: hit["lexical_score"], reverse=True)
    return ranked_hits


def _yo_variants(text: str) -> list[str]:
    swaps = str.maketrans("еёЕЁ", "ёеЁЕ")
    variants = [text, text.translate(swaps)]
    for index, character in enumerate(text):
        swapped = character.translate(swaps)
        if swapped != character:
            variants.append(text[:index] + swapped + text[index + 1 :])
    return list(dict.fromkeys(variants))


def _search_bm25(query: str, top_k: int) -> list[dict[str, Any]]:
    return VectorSearch().search_bm25(query, top_k=top_k)


@tool
def lexical_search_tool(query: str) -> list[dict[str, Any]]:
    """Lexical (keyword) search over documentation files using ripgrep.
    Expands the query with technical synonyms before searching.
    Returns matching snippets with metadata."""
    try:
        variants = expand_query_cached(query)
        return _run_lexical_search(variants)
    except Exception:
        return []


@tool
def vector_search_tool(query: str) -> list[dict[str, Any]]:
    """Semantic (vector) search over documentation chunks.
    Returns semantically similar chunks with metadata."""
    try:
        return VectorSearch().search(query, top_k=settings.fusion_top_k)
    except Exception:
        return []


# ─── Graph nodes ──────────────────────────────────────────────────────────────


async def normalize_query(state: GraphState, *, writer: StreamWriter) -> GraphState:
    query = state["query"].strip()
    writer(("node_start", {"node": "normalize_query", "query": query}))
    writer(("node_result", {"node": "normalize_query"}))
    return {"query": query}


async def run_vector_search(state: GraphState, *, writer: StreamWriter) -> GraphState:
    """Node: semantic search via ChromaDB."""
    writer(("node_start", {"node": "vector_search", "query": state["query"]}))
    result = vector_search_tool.invoke({"query": state["query"]})
    hits = result if isinstance(result, list) else []
    writer(
        (
            "node_result",
            {"node": "vector_search", "hits_count": len(hits), "hits": _summarize_hits(hits)},
        )
    )
    return {"vector_hits": hits}


async def extract_keywords_from_vector(state: GraphState, *, writer: StreamWriter) -> GraphState:
    """Node: extract key terms from vector search results."""
    writer(("node_start", {"node": "extract_keywords_from_vector"}))
    if not state["vector_hits"]:
        writer(
            (
                "node_result",
                {"node": "extract_keywords_from_vector", "keywords": [], "note": "no vector hits"},
            )
        )
        return {"extracted_keywords": []}
    keywords = await asyncio.to_thread(extract_keywords, state["vector_hits"])
    writer(("node_result", {"node": "extract_keywords_from_vector", "keywords": keywords}))
    return {"extracted_keywords": keywords}


async def run_bm25_search(state: GraphState, *, writer: StreamWriter) -> GraphState:
    """Node: full-corpus BM25 search using the query and vector-derived terms."""
    query_terms = [state["query"], *state.get("extracted_keywords", [])]
    bm25_query = " ".join(query_terms)
    writer(("node_start", {"node": "bm25_search", "query": bm25_query}))
    hits = await asyncio.to_thread(_search_bm25, bm25_query, settings.fusion_top_k)
    writer(
        (
            "node_result",
            {
                "node": "bm25_search",
                "hits_count": len(hits),
                "hits": _summarize_hits(hits),
            },
        )
    )
    return {"bm25_hits": hits}


async def run_lexical_search_pass1(state: GraphState, *, writer: StreamWriter) -> GraphState:
    """Node: first lexical search pass — original query + synonyms + vector keywords."""
    # Base: LLM-expanded variants of the original query
    variants = expand_query_cached(state["query"])
    writer(("node_start", {"node": "lexical_search_pass1", "variants": variants}))

    # Enrich with keywords extracted from vector hits
    for kw in state.get("extracted_keywords", []):
        if kw.lower() not in (v.lower() for v in variants):
            variants.append(kw)

    hits = _run_lexical_search(variants)
    writer(
        (
            "node_result",
            {
                "node": "lexical_search_pass1",
                "hits_count": len(hits),
                "hits": _summarize_hits(hits),
            },
        )
    )
    return {"lexical_hits_pass1": hits}


async def extract_keywords_from_lexical(state: GraphState, *, writer: StreamWriter) -> GraphState:
    """Node: extract key terms from first lexical pass results."""
    writer(("node_start", {"node": "extract_keywords_from_lexical"}))
    if not state.get("lexical_hits_pass1"):
        writer(
            (
                "node_result",
                {
                    "node": "extract_keywords_from_lexical",
                    "keywords": [],
                    "note": "no lexical hits",
                },
            )
        )
        return {"lexical_keywords": []}
    keywords = await asyncio.to_thread(extract_keywords, state["lexical_hits_pass1"])
    writer(("node_result", {"node": "extract_keywords_from_lexical", "keywords": keywords}))
    return {"lexical_keywords": keywords}


async def run_lexical_search_pass2(state: GraphState, *, writer: StreamWriter) -> GraphState:
    """Node: second lexical search pass — new keywords from first pass."""
    new_keywords = state.get("lexical_keywords", [])
    writer(("node_start", {"node": "lexical_search_pass2", "keywords": new_keywords}))
    if not new_keywords:
        writer(
            (
                "node_result",
                {"node": "lexical_search_pass2", "hits_count": 0, "note": "no new keywords"},
            )
        )
        return {"lexical_hits_pass2": []}
    hits = _run_lexical_search(new_keywords)
    writer(
        (
            "node_result",
            {
                "node": "lexical_search_pass2",
                "hits_count": len(hits),
                "hits": _summarize_hits(hits),
            },
        )
    )
    return {"lexical_hits_pass2": hits}


async def fuse_results(state: GraphState, *, writer: StreamWriter) -> GraphState:
    """Node: weighted fusion of all three result sets."""
    writer(("node_start", {"node": "fuse_results"}))
    all_lexical = state.get("lexical_hits_pass1", []) + state.get("lexical_hits_pass2", [])
    lexical_hits = sorted(
        all_lexical,
        key=lambda hit: hit.get("lexical_score", 0.0),
        reverse=True,
    )
    fused_items = reciprocal_rank_fusion(
        [state.get("vector_hits", []), state.get("bm25_hits", []), lexical_hits],
        top_k=settings.final_top_k,
    )
    writer(
        (
            "node_result",
            {
                "node": "fuse_results",
                "fused_count": len(fused_items),
                "hits": _summarize_hits(fused_items),
            },
        )
    )
    return {"fused": fused_items}


async def generate_answer(state: GraphState, *, writer: StreamWriter) -> GraphState:
    """Node: generate answer via streaming LLM call.
    Tokens are streamed to the client via LangGraph's ``StreamWriter``.
    The full answer is accumulated and stored in state.

    Uses the OpenAI client directly (not LangChain's ChatOpenAI) to ensure
    proper handling of vendor-specific streaming behavior.

    For each chunk from the OpenAI stream, we emit a custom event via
    ``StreamWriter`` with answer content tokens.
    The ``routes.py`` endpoint uses ``stream_mode=["custom"]`` to receive
    these events and forward them to the client as SSE events.
    """
    writer(("node_start", {"node": "generate_answer"}))
    if not state["fused"]:
        writer(("node_result", {"node": "generate_answer", "note": "no fused results to answer"}))
        return {"answer": None}

    model_name = state.get("model") or settings.chat_model
    primary_source = state["fused"][0]
    primary_rel_path = primary_source.get("metadata", {}).get("rel_path")
    if not isinstance(primary_rel_path, str) or not primary_rel_path:
        raise ValueError("The highest-ranked search result has no documentation file path.")

    max_full_files = max(1, settings.full_file_max_files)
    loaded_rel_paths: set[str] = set()
    context_chunks: list[dict[str, Any]] = []
    for index, source in enumerate(state["fused"]):
        rel_path = source.get("metadata", {}).get("rel_path")
        if (
            isinstance(rel_path, str)
            and rel_path
            and rel_path not in loaded_rel_paths
            and len(loaded_rel_paths) < max_full_files
        ):
            document = await asyncio.to_thread(read_full_file, rel_path)
            loaded_rel_paths.add(rel_path)
            context_chunks.append(
                {
                    **source,
                    "text": document,
                    "metadata": {
                        **source.get("metadata", {}),
                        "full_document": True,
                        "highest_ranked_source": index == 0,
                    },
                }
            )
        elif isinstance(rel_path, str) and rel_path in loaded_rel_paths:
            continue
        else:
            context_chunks.append(source)

    prompt = build_prompt(state["query"], context_chunks, include_reasoning=False)

    client = AsyncOpenAI(
        base_url=settings.llm_api_base_url,
        api_key=settings.llm_api_key,
        timeout=300.0,
        max_retries=0,
    )
    full_content: list[str] = []
    full_content: list[str] = []

    stream = await client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        stream_options={"include_usage": False},
    )

    async for chunk in stream:
        delta = chunk.choices[0].delta if chunk.choices else None
        if delta is None:
            continue

        content_token: str | None = None

        # Regular content tokens
        if delta.content:
            content_token = delta.content
            full_content.append(content_token)

        # Emit answer tokens via StreamWriter; internal reasoning is never forwarded.
        if content_token:
            writer(("token", content_token))

    writer(("node_result", {"node": "generate_answer"}))
    return {"answer": "".join(full_content) or None}


# ─── Graph definition ─────────────────────────────────────────────────────────

graph = StateGraph(GraphState)
graph.add_node("normalize", normalize_query)
graph.add_node("vector_search", run_vector_search)
graph.add_node("extract_keywords_v", extract_keywords_from_vector)
graph.add_node("bm25_search", run_bm25_search)
graph.add_node("lexical_search_p1", run_lexical_search_pass1)
graph.add_node("extract_keywords_l", extract_keywords_from_lexical)
graph.add_node("lexical_search_p2", run_lexical_search_pass2)
graph.add_node("fuse", fuse_results)
graph.add_node("answer", generate_answer)

graph.set_entry_point("normalize")
graph.add_edge("normalize", "vector_search")
graph.add_edge("vector_search", "extract_keywords_v")
graph.add_edge("extract_keywords_v", "bm25_search")
graph.add_edge("bm25_search", "lexical_search_p1")
graph.add_edge("lexical_search_p1", "extract_keywords_l")
graph.add_edge("extract_keywords_l", "lexical_search_p2")
graph.add_edge("lexical_search_p2", "fuse")
graph.add_edge("fuse", "answer")
graph.add_edge("answer", END)

app = graph.compile()
