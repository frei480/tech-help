from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any


_TOKEN_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)
_STOP_WORDS = {
    "а",
    "без",
    "бы",
    "в",
    "во",
    "вот",
    "все",
    "всего",
    "для",
    "до",
    "его",
    "ее",
    "ей",
    "если",
    "же",
    "за",
    "и",
    "из",
    "или",
    "их",
    "к",
    "как",
    "ко",
    "ли",
    "мне",
    "на",
    "над",
    "не",
    "него",
    "нее",
    "ни",
    "но",
    "о",
    "об",
    "от",
    "по",
    "под",
    "при",
    "с",
    "со",
    "так",
    "то",
    "у",
    "что",
    "чтобы",
    "это",
    "эта",
    "эти",
    "the",
    "and",
    "for",
    "from",
    "in",
    "of",
    "on",
    "or",
    "the",
    "to",
}


def _tokenize(text: str) -> list[str]:
    tokens = [match.group(0).casefold().replace("ё", "е") for match in _TOKEN_PATTERN.finditer(text)]
    return [token for token in tokens if len(token) > 1 and token not in _STOP_WORDS]


class BM25Index:
    def __init__(
        self,
        documents: list[dict[str, Any]],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.tokenized_documents = [_tokenize(item.get("text", "")) for item in documents]
        self.document_count = len(self.tokenized_documents)
        self.average_length = (
            sum(map(len, self.tokenized_documents)) / self.document_count
            if self.document_count
            else 0
        )
        self.document_frequencies: Counter[str] = Counter()
        self.term_frequencies: list[Counter[str]] = []
        for tokens in self.tokenized_documents:
            frequencies = Counter(tokens)
            self.term_frequencies.append(frequencies)
            self.document_frequencies.update(frequencies.keys())

    def search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        """Rank indexed documentation chunks with BM25Okapi."""
        if not self.documents or top_k <= 0 or self.average_length == 0:
            return []

        query_terms = set(_tokenize(query))
        if not query_terms:
            return []

        inverse_document_frequencies = {
            term: math.log1p(
                (self.document_count - self.document_frequencies[term] + 0.5)
                / (self.document_frequencies[term] + 0.5)
            )
            for term in query_terms
            if self.document_frequencies[term]
        }

        scored: list[tuple[float, int]] = []
        for index, frequencies in enumerate(self.term_frequencies):
            length = len(self.tokenized_documents[index])
            score = 0.0
            for term, inverse_document_frequency in inverse_document_frequencies.items():
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                normalization = frequency + self.k1 * (
                    1 - self.b + self.b * length / self.average_length
                )
                score += (
                    inverse_document_frequency
                    * frequency
                    * (self.k1 + 1)
                    / normalization
                )
            if score > 0:
                scored.append((score, index))

        scored.sort(key=lambda result: result[0], reverse=True)
        return [
            {**self.documents[index], "bm25_score": score}
            for score, index in scored[:top_k]
        ]


def rank_documents(
    query: str,
    documents: list[dict[str, Any]],
    top_k: int = 10,
    *,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[dict[str, Any]]:
    return BM25Index(documents, k1=k1, b=b).search(query, top_k=top_k)
