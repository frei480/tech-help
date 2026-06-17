import uuid
from typing import Any


import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import settings
from app.ingestion.chunker import Chunk


class ChromaStore:
    def __init__(self) -> None:
        self.client = chromadb.PersistentClient(
            path=str(settings.chroma_persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        import httpx

        response = httpx.post(
            f"{settings.openai_api_base_url}/embeddings",
            headers={
                "Authorization": f"Bearer {settings.openai_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": settings.embedding_model,
                "input": texts,
                "dimensions": settings.embedding_dimensions,
            },
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

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        return len(chunks)

    def search(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        top_k = top_k or settings.fusion_top_k
        query_embedding = self._embed_texts([query])[0]
        results = self.collection.query(
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
