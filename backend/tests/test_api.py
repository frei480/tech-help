import json
from collections.abc import AsyncIterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.api.routes import app

client = TestClient(app)


def test_ingest_missing_root() -> None:
    response = client.post("/api/ingest")
    assert response.status_code == 200
    payload = response.json()
    assert "indexed_files" in payload


def test_search_stream_keeps_fused_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.api.routes as routes

    class FakeGraph:
        async def astream(
            self, initial_state: dict[str, Any], stream_mode: list[str]
        ) -> AsyncIterator[tuple[str, dict[str, Any]]]:
            yield "updates", {
                "fuse": {
                    "fused": [
                        {
                            "text": "IIS HTTPS instructions",
                            "metadata": {"rel_path": "iis.txt", "chunk_index": 0},
                            "score": 0.9,
                        }
                    ]
                }
            }
            yield "updates", {
                "answer": {
                    "answer": "Configure IIS to use HTTPS.",
                    "reasoning_content": None,
                }
            }

    monkeypatch.setattr(routes, "graph_app", FakeGraph())

    response = client.post("/api/search/stream", json={"query": "IIS HTTPS"})

    assert response.status_code == 200
    done_event = response.text.split("event: done\ndata: ", maxsplit=1)[1]
    payload = json.loads(done_event.split("\n\n", maxsplit=1)[0])
    assert payload["answer"] == "Configure IIS to use HTTPS."
    assert payload["chunks"][0]["metadata"]["rel_path"] == "iis.txt"


def test_llm_settings_can_be_read_and_updated(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.config as config

    monkeypatch.setattr(config.settings, "llm_provider", "lm_studio")
    monkeypatch.setattr(config.settings, "chat_model", "local-chat")
    monkeypatch.setattr(config.settings, "embedding_model", "local-embedding")

    response = client.put(
        "/api/llm/settings",
        json={
            "provider": "openrouter",
            "chat_model": "openrouter-chat",
            "embedding_model": "openrouter-embedding",
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "provider": "openrouter",
        "chat_model": "openrouter-chat",
        "embedding_model": "openrouter-embedding",
    }

    response = client.get("/api/llm/settings")
    assert response.json()["provider"] == "openrouter"
