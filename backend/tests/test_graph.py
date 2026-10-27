from collections.abc import AsyncIterator
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
async def test_generate_answer_emits_node_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def stream() -> AsyncIterator[Any]:
        yield SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content="Generated answer"))]
        )

    class FakeCompletions:
        async def create(self, **kwargs: Any) -> AsyncIterator[Any]:
            return stream()

    monkeypatch.setattr(
        graph,
        "AsyncOpenAI",
        lambda **kwargs: SimpleNamespace(
            chat=SimpleNamespace(completions=FakeCompletions())
        ),
    )
    events: list[tuple[str, Any]] = []

    result = await graph.generate_answer(
        {
            "query": "test query",
            "mode": "ai_answer",
            "model": None,
            "fused": [{"text": "source"}],
        },
        writer=events.append,
    )

    assert result["answer"] == "Generated answer"
    assert ("node_result", {"node": "generate_answer"}) in events
