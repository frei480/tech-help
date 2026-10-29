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
    monkeypatch.setattr(graph, "read_full_file", lambda rel_path: "Complete source document.")
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
            ],
        },
        writer=events.append,
    )

    assert result["answer"] == "Generated answer"
    assert "reasoning_content" not in result
    assert ("node_result", {"node": "generate_answer"}) in events
    assert not any("This must remain private." in str(event) for event in events)
    prompt = request["messages"][0]["content"]
    assert "[guide.md:2] (complete document; highest-ranked source)\nComplete source document." in prompt
    assert "matched excerpt" not in prompt
    assert "[other.md:0]\nsecondary source" in prompt
    assert "split multi-part questions" in prompt
    assert "you do not have web search here" in prompt
    assert "Give a complete, detailed answer to every part of the question" in prompt
    assert "do not omit relevant documented details merely" in prompt
    assert "Give the most complete and" in prompt
    assert "Answer in the same language as the user's question." in prompt
    assert "Do not reveal internal reasoning" in prompt
