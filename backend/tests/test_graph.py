import asyncio
from collections.abc import AsyncIterator
from threading import Event
from types import SimpleNamespace
from typing import Any

import pytest

from app import graph


@pytest.mark.asyncio
async def test_normalize_query_emits_node_result() -> None:
    events: list[tuple[str, Any]] = []

    result = await graph.normalize_query(
        {
            "query": "  test query  ",
            "mode": "ai_answer",
            "model": None,
        },
        writer=events.append,
    )

    assert result["query"] == "test query"
    assert ("node_result", {"node": "normalize_query"}) in events


@pytest.mark.asyncio
async def test_vector_keyword_extraction_emits_running_event_before_finishing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extraction_started = Event()
    finish_extraction = Event()
    events: list[tuple[str, Any]] = []

    def wait_for_extraction() -> list[str]:
        extraction_started.set()
        assert finish_extraction.wait(timeout=5)
        return ["keyword"]

    monkeypatch.setattr(graph, "extract_keywords", lambda chunks: wait_for_extraction())

    task = asyncio.create_task(
        graph.extract_keywords_from_vector(
            {
                "query": "test query",
                "mode": "ai_answer",
                "model": None,
                "vector_hits": [{"text": "source"}],
            },
            writer=events.append,
        )
    )
    try:
        await asyncio.sleep(0)
        assert extraction_started.wait(timeout=5)
        assert ("node_start", {"node": "extract_keywords_from_vector"}) in events
        assert not task.done()
    finally:
        finish_extraction.set()

    result = await task
    assert result["extracted_keywords"] == ["keyword"]
    assert ("node_result", {"node": "extract_keywords_from_vector", "keywords": ["keyword"]}) in events


@pytest.mark.asyncio
async def test_bm25_search_uses_question_and_vector_keywords(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    events: list[tuple[str, Any]] = []

    class FakeVectorSearch:
        def search_bm25(self, query: str, top_k: int) -> list[dict[str, Any]]:
            captured["query"] = query
            captured["top_k"] = top_k
            return [{"text": "architecture", "metadata": {"rel_path": "architecture.txt"}}]

    monkeypatch.setattr(graph, "VectorSearch", FakeVectorSearch)

    result = await graph.run_bm25_search(
        {
            "query": "Which architecture?",
            "mode": "ai_answer",
            "model": None,
            "vector_hits": [],
            "extracted_keywords": ["three-tier architecture"],
        },
        writer=events.append,
    )

    assert captured["query"] == "Which architecture? three-tier architecture"
    assert result["bm25_hits"][0]["metadata"]["rel_path"] == "architecture.txt"
    assert ("node_result", {"node": "bm25_search", "hits_count": 1, "hits": [
        {
            "rel_path": "architecture.txt",
            "line_number": None,
            "score": None,
            "snippet_preview": "architecture",
        }
    ]}) in events


def test_lexical_search_ranks_specific_terms_above_broad_matches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_ripgrep(query: str) -> list[SimpleNamespace]:
        if query == "T-FLEX DOCs":
            return [
                SimpleNamespace(
                    rel_path="release-notes.txt",
                    line_number=1,
                    snippet="T-FLEX DOCs release notes",
                    file_path="release-notes.txt",
                ),
                SimpleNamespace(
                    rel_path="architecture.txt",
                    line_number=2,
                    snippet="T-FLEX DOCs uses a three-tier architecture",
                    file_path="architecture.txt",
                ),
            ]
        if query == "трёхуровневая архитектура":
            return [
                SimpleNamespace(
                    rel_path="architecture.txt",
                    line_number=2,
                    snippet="T-FLEX DOCs uses a three-tier architecture",
                    file_path="architecture.txt",
                )
            ]
        return []

    monkeypatch.setattr(graph, "run_ripgrep", fake_ripgrep)

    hits = graph._run_lexical_search(["T-FLEX DOCs", "трёхуровневая архитектура"])

    assert hits[0]["metadata"]["rel_path"] == "architecture.txt"
    assert hits[0]["lexical_score"] == 1
    assert hits[1]["metadata"]["rel_path"] == "release-notes.txt"
    assert hits[1]["lexical_score"] < hits[0]["lexical_score"]


def test_lexical_search_matches_e_and_yo_spellings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    searched_queries: list[str] = []

    def fake_ripgrep(query: str) -> list[SimpleNamespace]:
        searched_queries.append(query)
        if query == "трёхуровневая архитектура":
            return [
                SimpleNamespace(
                    rel_path="architecture.txt",
                    line_number=2,
                    snippet="В основе лежит трёхуровневая архитектура",
                    file_path="architecture.txt",
                )
            ]
        return []

    monkeypatch.setattr(graph, "run_ripgrep", fake_ripgrep)

    hits = graph._run_lexical_search(["трехуровневая архитектура"])

    assert "трехуровневая архитектура" in searched_queries
    assert "трёхуровневая архитектура" in searched_queries
    assert len(hits) == 1
    assert hits[0]["metadata"]["rel_path"] == "architecture.txt"


@pytest.mark.asyncio
async def test_fuse_results_keeps_only_one_hit_per_document() -> None:
    events: list[tuple[str, Any]] = []
    result = await graph.fuse_results(
        {
            "query": "test query",
            "mode": "ai_answer",
            "model": None,
            "vector_hits": [],
            "extracted_keywords": [],
            "lexical_hits_pass1": [
                {
                    "text": "first excerpt",
                    "metadata": {"rel_path": "Guide.md", "line_number": 1},
                    "lexical_score": 1.0,
                },
                {
                    "text": "second excerpt",
                    "metadata": {"rel_path": "guide.md", "line_number": 2},
                    "lexical_score": 0.9,
                },
                {
                    "text": "other document",
                    "metadata": {"rel_path": "other.md", "line_number": 1},
                    "lexical_score": 0.8,
                },
            ],
            "lexical_hits_pass2": [],
        },
        writer=events.append,
    )

    assert [hit["metadata"]["rel_path"] for hit in result["fused"]] == [
        "Guide.md",
        "other.md",
    ]


@pytest.mark.asyncio
async def test_generate_answer_emits_node_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request: dict[str, Any] = {}

    async def stream() -> AsyncIterator[Any]:
        yield SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        content="Generated answer",
                        reasoning_content="This must remain private.",
                    )
                )
            ]
        )

    class FakeCompletions:
        async def create(self, **kwargs: Any) -> AsyncIterator[Any]:
            request.update(kwargs)
            return stream()

    monkeypatch.setattr(
        graph,
        "AsyncOpenAI",
        lambda **kwargs: SimpleNamespace(
            chat=SimpleNamespace(completions=FakeCompletions())
        ),
    )
    monkeypatch.setattr(
        graph.settings,
        "full_file_max_files",
        3,
    )
    loaded_paths: list[str] = []

    def read_document(rel_path: str) -> str:
        loaded_paths.append(rel_path)
        return f"Complete document: {rel_path}"

    monkeypatch.setattr(graph, "read_full_file", read_document)
    events: list[tuple[str, Any]] = []

    result = await graph.generate_answer(
        {
            "query": "test query",
            "mode": "ai_answer",
            "model": None,
            "fused": [
                {
                    "text": "matched excerpt",
                    "metadata": {"rel_path": "guide.md", "chunk_index": 2},
                },
                {"text": "secondary source", "metadata": {"rel_path": "other.md"}},
                {"text": "another excerpt", "metadata": {"rel_path": "guide.md", "chunk_index": 3}},
                {
                    "text": "architecture excerpt",
                    "metadata": {"rel_path": "architecture.md", "chunk_index": 0},
                },
                {"text": "fourth document excerpt", "metadata": {"rel_path": "extra.md"}},
            ],
        },
        writer=events.append,
    )

    assert result["answer"] == "Generated answer"
    assert "reasoning_content" not in result
    assert ("node_result", {"node": "generate_answer"}) in events
    assert not any("This must remain private." in str(event) for event in events)
    prompt = request["messages"][0]["content"]
    assert (
        "[guide.md:2] (complete document; highest-ranked source)\n"
        "Complete document: guide.md"
    ) in prompt
    assert "matched excerpt" not in prompt
    assert "[other.md:0] (complete document)\nComplete document: other.md" in prompt
    assert "another excerpt" not in prompt
    assert (
        "[architecture.md:0] (complete document)\n"
        "Complete document: architecture.md"
    ) in prompt
    assert "[extra.md:0]\nfourth document excerpt" in prompt
    assert loaded_paths == ["guide.md", "other.md", "architecture.md"]
    assert "split multi-part questions" in prompt
    assert "you do not have web search here" in prompt
    assert "Give a complete, detailed answer to every part of the question" in prompt
    assert "do not omit relevant documented details merely" in prompt
    assert "Give the most complete and" in prompt
    assert "Answer in the same language as the user's question." in prompt
    assert "Do not reveal internal reasoning" in prompt
