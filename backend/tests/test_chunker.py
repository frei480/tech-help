from pathlib import Path

from app.ingestion.chunker import chunk_text


def test_chunk_text_basic() -> None:
    text = "word " * 500
    path = Path("demo.txt")
    chunks = chunk_text(text, path, "demo.txt")
    assert len(chunks) > 0
    for chunk in chunks:
        assert len(chunk.text) > 0
        assert chunk.file_path == path
        assert chunk.rel_path == "demo.txt"
