from typing import Any


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
        chunks: list[dict[str, Any]],
    ) -> None:
        self.query = query
        self.mode = mode
        self.answer = answer
        self.chunks = chunks
