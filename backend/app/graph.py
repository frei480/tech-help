import json
from typing import Any, TypedDict

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph


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
    reasoning_summary: list[str]
    source_notes: list[dict[str, Any]]
    limitations: list[str]


# ─── Helpers ──────────────────────────────────────────────────────────────────


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


async def normalize_query(state: GraphState) -> GraphState:
    return {"query": state["query"].strip()}


async def run_vector_search(state: GraphState) -> GraphState:
    """Node: semantic search via ChromaDB."""
    result = vector_search_tool.invoke({"query": state["query"]})
    return {"vector_hits": result if isinstance(result, list) else []}


async def extract_keywords_from_vector(state: GraphState) -> GraphState:
    """Node: extract key terms from vector search results."""
    if not state["vector_hits"]:
        return {"extracted_keywords": []}
    keywords = extract_keywords(state["vector_hits"])
    return {"extracted_keywords": keywords}


async def run_lexical_search_pass1(state: GraphState) -> GraphState:
    """Node: first lexical search pass — original query + synonyms + vector keywords."""
    # Base: LLM-expanded variants of the original query
    variants = expand_query_cached(state["query"])

    # Enrich with keywords extracted from vector hits
    for kw in state.get("extracted_keywords", []):
        if kw.lower() not in (v.lower() for v in variants):
            variants.append(kw)

    hits = _run_lexical_search(variants)
    return {"lexical_hits_pass1": hits}


async def extract_keywords_from_lexical(state: GraphState) -> GraphState:
    """Node: extract key terms from first lexical pass results."""
    if not state.get("lexical_hits_pass1"):
        return {"lexical_keywords": []}
    keywords = extract_keywords(state["lexical_hits_pass1"])
    return {"lexical_keywords": keywords}


async def run_lexical_search_pass2(state: GraphState) -> GraphState:
    """Node: second lexical search pass — new keywords from first pass."""
    new_keywords = state.get("lexical_keywords", [])
    if not new_keywords:
        return {"lexical_hits_pass2": []}
    hits = _run_lexical_search(new_keywords)
    return {"lexical_hits_pass2": hits}


async def fuse_results(state: GraphState) -> GraphState:
    """Node: weighted fusion of all three result sets."""
    all_lexical = state.get("lexical_hits_pass1", []) + state.get("lexical_hits_pass2", [])
    fused = weighted_fusion(
        lexical_hits=all_lexical,
        vector_hits=state["vector_hits"],
        top_k=settings.final_top_k,
    )
    return {"fused": [item["item"] for item in fused]}


async def generate_answer(state: GraphState) -> GraphState:
    if not state["fused"]:
        return {"answer": None}

    model_name = state.get("model") or settings.chat_model
    _llm = ChatOpenAI(
        base_url=settings.openai_api_base_url,
        api_key=settings.openai_api_key,
        model=model_name,
        timeout=300.0,
        streaming=True,
    ).bind_tools(_llm_tools)

    prompt = build_prompt(state["query"], state["fused"], include_reasoning=False)
    result = await _llm.ainvoke([{"role": "user", "content": prompt}])
    return {"answer": result.content}


async def generate_reasoning(state: GraphState) -> GraphState:
    if not state["fused"] or not state.get("answer"):
        return {"reasoning_summary": [], "source_notes": [], "limitations": []}

    model_name = state.get("model") or settings.chat_model
    _llm = ChatOpenAI(
        base_url=settings.openai_api_base_url,
        api_key=settings.openai_api_key,
        model=model_name,
        timeout=300.0,
        streaming=True,
    )
    prompt = build_prompt(state["query"], state["fused"], include_reasoning=True)
    messages = [{"role": "user", "content": state["answer"]}, {"role": "user", "content": prompt}]

    try:
        result = await _llm.ainvoke(messages)
        text = str(result.content)
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            parsed = json.loads(text[start : end + 1])
            return {
                "reasoning_summary": parsed.get("reasoning_summary", []),
                "source_notes": parsed.get("source_notes", []),
                "limitations": parsed.get("limitations", []),
            }
    except Exception:
        pass

    return {"reasoning_summary": [], "source_notes": [], "limitations": []}


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
graph.add_node("reasoning", generate_reasoning)

graph.set_entry_point("normalize")
graph.add_edge("normalize", "vector_search")
graph.add_edge("vector_search", "extract_keywords_v")
graph.add_edge("extract_keywords_v", "lexical_search_p1")
graph.add_edge("lexical_search_p1", "extract_keywords_l")
graph.add_edge("extract_keywords_l", "lexical_search_p2")
graph.add_edge("lexical_search_p2", "fuse")
graph.add_edge("fuse", "answer")
graph.add_edge("answer", "reasoning")
graph.add_edge("reasoning", END)

app = graph.compile()
