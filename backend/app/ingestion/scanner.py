from pathlib import Path

from app.config import settings


def scan_docs(root: Path | None = None) -> list[Path]:
    root = root or settings.docs_root
    if not root.exists():
        return []
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in {".txt", ".md"}
    )
