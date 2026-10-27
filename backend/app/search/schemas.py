from typing import TypedDict


class SearchChunkMetadata(TypedDict):
    file_path: str
    rel_path: str
    chunk_index: int
    start_token: int
    end_token: int
    heading: str


class SearchChunk(TypedDict):
    text: str
    metadata: SearchChunkMetadata


class SearchRequest:
    def __init__(self, query: str, mode: str = "ai_answer") -> None:
        self.query = query
        self.mode = mode


class SearchResponse:
    def __init__(
        self,
        query: str,
        mode: str,
        answer: str | None,
        chunks: list[SearchChunk],
    ) -> None:
        self.query = query
        self.mode = mode
        self.answer = answer
        self.chunks = chunks
