from typing import Any, TypedDict

from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from langgraph.types import StreamWriter
from openai import AsyncOpenAI


from app.config import settings
from app.llm.prompts import build_prompt
from app.search.file_loader import load_full_file_tool
from app.search.fusion import weighted_fusion
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
    lexical_hits_pass1: list[dict[str, Any]]
    lexical_keywords: list[str]
    lexical_hits_pass2: list[dict[str, Any]]
    fused: list[dict[str, Any]]
    answer: str | None
    reasoning_content: str | None


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
                "score": h.get("lexical_score") or h.get("score"),
                "snippet_preview": (h.get("text", "") or "")[:200],
            }
        )
    return out


def _run_lexical_search(variants: list[str]) -> list[dict[str, Any]]:
    """Run ripgrep for each query variant and return deduplicated hits."""
    all_hits: list[dict[str, Any]] = []
    seen: set[str] = set()

    for variant in variants:
        hits = run_ripgrep(variant)
        for hit in hits:
            key = f"{hit.rel_path}:{hit.line_number}"
            if key not in seen:
                seen.add(key)
                all_hits.append(
                    {
                        "text": hit.snippet,
                        "metadata": {
                            "file_path": str(hit.file_path),
                            "rel_path": hit.rel_path,
                            "line_number": hit.line_number,
                            "chunk_index": -1,
                        },
                        "lexical_score": 1.0,
                    }
                )
    return all_hits


# ─── Tools ────────────────────────────────────────────────────────────────────


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


# Tools that the answer-generation LLM can invoke on its own
_llm_tools = [load_full_file_tool]


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
    keywords = extract_keywords(state["vector_hits"])
    writer(("node_result", {"node": "extract_keywords_from_vector", "keywords": keywords}))
    return {"extracted_keywords": keywords}


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
    keywords = extract_keywords(state["lexical_hits_pass1"])
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
    fused = weighted_fusion(
        lexical_hits=all_lexical,
        vector_hits=state["vector_hits"],
        top_k=settings.final_top_k,
    )
    fused_items = [item["item"] for item in fused]
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
    Reasoning content (chain-of-thought) from reasoning models is also captured.

    Uses the OpenAI client directly (not LangChain's ChatOpenAI) to ensure
    proper handling of vendor-specific delta fields like ``reasoning_content``
    that LM Studio sends as a top-level field in the delta JSON.

    For each chunk from the OpenAI stream, we emit a custom event via
    ``StreamWriter`` with both ``content`` and ``reasoning_content`` tokens.
    The ``routes.py`` endpoint uses ``stream_mode=["custom"]`` to receive
    these events and forward them to the client as SSE events.
    """
    writer(("node_start", {"node": "generate_answer"}))
    if not state["fused"]:
        writer(("node_result", {"node": "generate_answer", "note": "no fused results to answer"}))
        return {"answer": None}

    model_name = state.get("model") or settings.chat_model
    prompt = build_prompt(state["query"], state["fused"], include_reasoning=False)

    client = AsyncOpenAI(
        base_url=settings.llm_api_base_url,
        api_key=settings.llm_api_key,
        timeout=300.0,
        max_retries=0,
    )

    full_content: list[str] = []
    full_reasoning: list[str] = []

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
        reasoning_token: str | None = None

        # Regular content tokens
        if delta.content:
            content_token = delta.content
            full_content.append(content_token)

        # Reasoning content tokens (vendor-specific field from LM Studio)
        # This is a top-level field in the delta JSON. The OpenAI SDK's Delta
        # object may expose it as a direct attribute or via model_extra.
        reasoning = getattr(delta, "reasoning_content", None)
        if reasoning is None:
            reasoning = (
                (delta.model_extra or {}).get("reasoning_content")
                if hasattr(delta, "model_extra")
                else None
            )
        if isinstance(reasoning, str) and reasoning:
            reasoning_token = reasoning
            full_reasoning.append(reasoning_token)

        # Emit custom event via StreamWriter for real-time streaming.
        # The routes.py endpoint picks these up via stream_mode=["custom"].
        if content_token:
            writer(("token", content_token))
        if reasoning_token:
            writer(("reasoning_token", reasoning_token))

    writer(("node_result", {"node": "generate_answer"}))
    return {
        "answer": "".join(full_content) or None,
        "reasoning_content": "".join(full_reasoning) or None,
    }


# ─── Graph definition ─────────────────────────────────────────────────────────

graph = StateGraph(GraphState)
graph.add_node("normalize", normalize_query)
graph.add_node("vector_search", run_vector_search)
graph.add_node("extract_keywords_v", extract_keywords_from_vector)
graph.add_node("lexical_search_p1", run_lexical_search_pass1)
graph.add_node("extract_keywords_l", extract_keywords_from_lexical)
graph.add_node("lexical_search_p2", run_lexical_search_pass2)
graph.add_node("fuse", fuse_results)
graph.add_node("answer", generate_answer)

graph.set_entry_point("normalize")
graph.add_edge("normalize", "vector_search")
graph.add_edge("vector_search", "extract_keywords_v")
graph.add_edge("extract_keywords_v", "lexical_search_p1")
graph.add_edge("lexical_search_p1", "extract_keywords_l")
graph.add_edge("extract_keywords_l", "lexical_search_p2")
graph.add_edge("lexical_search_p2", "fuse")
graph.add_edge("fuse", "answer")
graph.add_edge("answer", END)

app = graph.compile()
