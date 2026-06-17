import os
import re
from pathlib import Path
from typing import NamedTuple

import tiktoken

from app.config import settings

os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(settings.tiktoken_cache_dir))
settings.tiktoken_cache_dir.mkdir(parents=True, exist_ok=True)

enc = tiktoken.get_encoding("o200k_base")


class Chunk(NamedTuple):
    text: str
    file_path: Path
    rel_path: str
    chunk_index: int
    start_token: int
    end_token: int
    heading: str | None


def _extract_headings(text: str) -> str | None:
    match = re.search(r"^#{1,6}\s+.+$", text, re.MULTILINE)
    return match.group(0) if match else None


def chunk_text(text: str, file_path: Path, rel_path: str) -> list[Chunk]:
    tokens = enc.encode(text)
    chunks: list[Chunk] = []
    start = 0
    index = 0

    while start < len(tokens):
        end = min(start + settings.chunk_size_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text_str = enc.decode(chunk_tokens)
        heading = _extract_headings(chunk_text_str)
        chunks.append(
            Chunk(
                text=chunk_text_str,
                file_path=file_path,
                rel_path=rel_path,
                chunk_index=index,
                start_token=start,
                end_token=end,
                heading=heading,
            )
        )
        start += settings.chunk_size_tokens - settings.chunk_overlap_tokens
        index += 1

    return chunks
