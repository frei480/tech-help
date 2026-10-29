import json
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from threading import Event
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


def test_status_reports_ingestion_running(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.api.routes as routes

    ingest_started = Event()
    finish_ingest = Event()

    def wait_for_finish() -> list:
        ingest_started.set()
        assert finish_ingest.wait(timeout=5)
        return []

    class FakeCollection:
        def get(self, include: list[str]) -> dict[str, list]:
            return {"metadatas": []}

    class FakeStore:
        collection = FakeCollection()

        def count(self) -> int:
            return 0

    monkeypatch.setattr(routes, "scan_docs", wait_for_finish)
    monkeypatch.setattr(routes, "ChromaStore", FakeStore)

    with ThreadPoolExecutor(max_workers=1) as executor:
        ingest_request = executor.submit(client.post, "/api/ingest")
        try:
            assert ingest_started.wait(timeout=5)
            response = client.get("/api/status")
            assert response.status_code == 200
            assert response.json()["is_indexing"] is True
        finally:
            finish_ingest.set()

        assert ingest_request.result(timeout=5).status_code == 200

    response = client.get("/api/status")
    assert response.json()["is_indexing"] is False


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
                    "reasoning_content": "This must not be sent to clients.",
                }
            }
            yield "custom", ("reasoning_token", "Private reasoning token.")

    monkeypatch.setattr(routes, "graph_app", FakeGraph())

    response = client.post("/api/search/stream", json={"query": "IIS HTTPS"})

    assert response.status_code == 200
    done_event = response.text.split("event: done\ndata: ", maxsplit=1)[1]
    payload = json.loads(done_event.split("\n\n", maxsplit=1)[0])
    assert payload["answer"] == "Configure IIS to use HTTPS."
    assert payload["chunks"][0]["metadata"]["rel_path"] == "iis.txt"
    assert "reasoning_content" not in payload
    assert "reasoning_token" not in response.text
    assert "Private reasoning token." not in response.text
    assert "This must not be sent to clients." not in response.text


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


def test_openrouter_embedding_models_use_embeddings_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.api.llm_router as llm_router
    import app.config as config

    requests: list[tuple[str, dict[str, str]]] = []

    class FakeResponse:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict[str, list[dict[str, str]]]:
            return {
                "data": [
                    {"id": "provider/embed-model", "name": "Embed Model"},
                    {"id": "provider/second-embed-model"},
                ]
            }

    class FakeAsyncClient:
        def __init__(self, timeout: float) -> None:
            assert timeout == 10.0

        async def __aenter__(self) -> "FakeAsyncClient":
            return self

        async def __aexit__(self, *args: object) -> None:
            pass

        async def get(self, url: str, headers: dict[str, str]) -> FakeResponse:
            requests.append((url, headers))
            return FakeResponse()

    monkeypatch.setattr(config.settings, "openrouter_api_base_url", "https://openrouter.ai/api/v1/")
    monkeypatch.setattr(config.settings, "openrouter_api_key", "test-key")
    monkeypatch.setattr(llm_router.httpx, "AsyncClient", FakeAsyncClient)

    response = client.get("/api/llm/embedding-models?provider=openrouter")

    assert response.status_code == 200
    assert response.json() == [
        {"id": "provider/embed-model", "name": "Embed Model"},
        {"id": "provider/second-embed-model", "name": "provider/second-embed-model"},
    ]
    assert requests == [
        (
            "https://openrouter.ai/api/v1/embeddings/models",
            {"Authorization": "Bearer test-key"},
        )
    ]
