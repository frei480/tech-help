from typing import Any

from app.ingestion.chroma_store import ChromaStore


class VectorSearch:
    def __init__(self) -> None:
        self.store = ChromaStore()

    def search(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        return self.store.search(query, top_k=top_k)
