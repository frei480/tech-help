import pytest
from chromadb.errors import InvalidArgumentError

from app.config import settings
from app.ingestion.chroma_store import ChromaStore


class DummyResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self.payload


def test_embedding_request_uses_vector_returned_by_service(monkeypatch: pytest.MonkeyPatch) -> None:
    store = ChromaStore.__new__(ChromaStore)

    monkeypatch.setattr(type(settings), "llm_api_base_url", property(lambda self: "https://example.test/v1"))
    monkeypatch.setattr(type(settings), "llm_api_key", property(lambda self: "token-123"))
    monkeypatch.setattr(settings, "embedding_model", "text-embedding-embeddinggemma-300m")

    captured: dict[str, object] = {}

    def fake_post(url: str, headers: dict[str, str], json: dict[str, object], timeout: float) -> DummyResponse:
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        captured["timeout"] = timeout
        return DummyResponse({"data": [{"embedding": [0.1, 0.2]}]})

    monkeypatch.setattr("httpx.post", fake_post)

    assert store._embed_texts(["hello"]) == [[0.1, 0.2]]
    assert captured["headers"] == {"Content-Type": "application/json", "Authorization": "Bearer token-123"}
    assert "dimensions" not in captured["json"]


def test_embedding_request_does_not_override_service_vector_dimensions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ChromaStore.__new__(ChromaStore)

    monkeypatch.setattr(type(settings), "llm_api_base_url", property(lambda self: "https://example.test/v1"))
    monkeypatch.setattr(type(settings), "llm_api_key", property(lambda self: ""))
    monkeypatch.setattr(settings, "embedding_model", "text-embedding-3-small")

    captured: dict[str, object] = {}
    embedding = [0.5, 0.6, 0.7]

    def fake_post(url: str, headers: dict[str, str], json: dict[str, object], timeout: float) -> DummyResponse:
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        captured["timeout"] = timeout
        return DummyResponse({"data": [{"embedding": embedding}]})

    monkeypatch.setattr("httpx.post", fake_post)

    embeddings = store._embed_texts(["hello"])

    assert embeddings == [embedding]
    assert len(embeddings[0]) == len(embedding)
    assert "dimensions" not in captured["json"]


def test_collection_reset_on_dimension_mismatch() -> None:
    store = ChromaStore.__new__(ChromaStore)
    first_calls = {"count": 0}
    second_calls = {"count": 0}

    class FirstCollection:
        def add(self, **kwargs: object) -> None:
            first_calls["count"] += 1
            raise InvalidArgumentError("Collection expecting embedding with dimension of 768, got 1024")

    class SecondCollection:
        def add(self, **kwargs: object) -> None:
            second_calls["count"] += 1

    first_collection = FirstCollection()
    second_collection = SecondCollection()
    store.collection = first_collection

    def fake_reset() -> None:
        store.collection = second_collection

    store._reset_collection = fake_reset

    store._add_with_collection_reset(
        ids=["doc-1"],
        embeddings=[[0.1, 0.2, 0.3]],
        documents=["hello"],
        metadatas=[{"rel_path": "test.txt"}],
    )

    assert first_calls["count"] == 1
    assert second_calls["count"] == 1


def test_bm25_search_reads_and_caches_indexed_documents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ChromaStore.__new__(ChromaStore)
    calls = {"count": 0}

    class Collection:
        def get(self, include: list[str]) -> dict[str, list]:
            calls["count"] += 1
            return {
                "documents": [
                    "T-FLEX DOCs provides an API.",
                    "T-FLEX DOCs uses a three-tier architecture.",
                ],
                "metadatas": [
                    {"rel_path": "api.txt"},
                    {"rel_path": "architecture.txt"},
                ],
            }

    store.collection = Collection()
    monkeypatch.setattr(ChromaStore, "_bm25_index", None)
    monkeypatch.setattr(ChromaStore, "_bm25_cache_key", None)
    monkeypatch.setattr(settings, "chroma_persist_dir", settings.chroma_persist_dir / "bm25-test")

    hits = store.search_bm25("three-tier architecture")
    cached_hits = store.search_bm25("three-tier architecture")

    assert hits[0]["metadata"]["rel_path"] == "architecture.txt"
    assert cached_hits == hits
    assert calls["count"] == 1
