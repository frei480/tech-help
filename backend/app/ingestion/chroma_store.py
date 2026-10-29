import uuid
from typing import Any


import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.errors import InvalidArgumentError

from app.config import settings
from app.ingestion.chunker import Chunk


class ChromaStore:
    def __init__(self) -> None:
        self.client = chromadb.PersistentClient(
            path=str(settings.chroma_persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self._get_or_create_collection()

    def _get_or_create_collection(self):
        return self.client.get_or_create_collection(
            name=settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def _reset_collection(self) -> None:
        try:
            self.client.delete_collection(name=settings.chroma_collection_name)
        except ValueError:
            pass
        self.collection = self._get_or_create_collection()

    def _add_with_collection_reset(self, **kwargs: Any) -> None:
        try:
            self.collection.add(**kwargs)
        except InvalidArgumentError as exc:
            message = str(exc)
            if "dimension" not in message.lower():
                raise
            self._reset_collection()
            self.collection.add(**kwargs)

    def _query_with_collection_reset(self, **kwargs: Any) -> dict[str, Any]:
        try:
            return self.collection.query(**kwargs)
        except InvalidArgumentError as exc:
            message = str(exc)
            if "dimension" not in message.lower():
                raise
            self._reset_collection()
            return self.collection.query(**kwargs)

    def _build_embedding_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        api_key = settings.llm_api_key.strip()
        if api_key:
            headers["Authorization"] = "Bearer " + api_key
        return headers

    def _build_embedding_payload(self, texts: list[str]) -> dict[str, Any]:
        return {
            "model": settings.embedding_model,
            "input": texts,
        }

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        import httpx

        response = httpx.post(
            f"{settings.llm_api_base_url}/embeddings",
            headers=self._build_embedding_headers(),
            json=self._build_embedding_payload(texts),
            timeout=60.0,
        )
        response.raise_for_status()
        data = response.json()
        return [item["embedding"] for item in data["data"]]

    def index_chunks(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0

        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{c.rel_path}:{c.chunk_index}")) for c in chunks]
        texts = [c.text for c in chunks]
        embeddings = self._embed_texts(texts)
        metadatas = [
            {
                "file_path": str(c.file_path),
                "rel_path": c.rel_path,
                "chunk_index": c.chunk_index,
                "start_token": c.start_token,
                "end_token": c.end_token,
                "heading": c.heading or "",
            }
            for c in chunks
        ]

        self._add_with_collection_reset(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        return len(chunks)

    def search(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        top_k = top_k or settings.fusion_top_k
        query_embedding = self._embed_texts([query])[0]
        results = self._query_with_collection_reset(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            hits.append(
                {
                    "text": doc,
                    "metadata": meta,
                    "vector_distance": dist,
                    "vector_score": max(0.0, 1.0 - dist),
                }
            )
        return hits

    def count(self) -> int:
        return self.collection.count()

    def delete_by_file(self, rel_path: str) -> None:
        matches = self.collection.get(
            where={"rel_path": rel_path},
            include=[],
        )
        if matches["ids"]:
            self.collection.delete(ids=matches["ids"])