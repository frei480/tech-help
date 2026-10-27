"""Load full documentation files — available as a tool for the LLM.

When the model needs more context than what individual chunks provide,
it can call ``load_full_file_tool`` with a ``rel_path`` to fetch the
complete file content.  The tool enforces a per-call token limit so
that very large files do not blow through the context window.
"""

import os
from pathlib import Path

import tiktoken
from langchain_core.tools import tool

from app.config import settings

os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(settings.tiktoken_cache_dir))
_enc = tiktoken.get_encoding("o200k_base")


def _count_tokens(text: str) -> int:
    return len(_enc.encode(text, disallowed_special=()))


@tool
def load_full_file_tool(rel_path: str) -> str:
    """Load the full content of a documentation file.

    Use this tool when the available chunks do not provide enough context
    and you need to see the complete document — including headings,
    structure, and related sections.

    Args:
        rel_path: Relative path of the file inside the docs directory
                  (e.g. ``"Справка по администрированию T-FLEX DOCs/01.htm.md"``).

    Returns:
        The full text of the file, or an error message if the file
        does not exist or cannot be read.
    """
    docs_root: Path = settings.docs_root
    file_path = docs_root / rel_path

    # Security: prevent path traversal outside docs_root
    try:
        file_path = file_path.resolve()
        docs_root_resolved = docs_root.resolve()
        if not str(file_path).startswith(str(docs_root_resolved)):
            return f"Error: access denied — path '{rel_path}' is outside the docs directory."
    except ValueError, OSError:
        return f"Error: invalid path '{rel_path}'."

    if not file_path.is_file():
        return f"Error: file '{rel_path}' not found."

    try:
        text = file_path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return f"Error reading '{rel_path}': {exc}"

    # Enforce a per-call token limit so the model does not exceed its window
    max_tokens = settings.full_file_max_tokens
    token_count = _count_tokens(text)
    if token_count > max_tokens:
        tokens = _enc.encode(text, disallowed_special=())
        text = _enc.decode(tokens[:max_tokens]) + "\n\n[… truncated …]"

    return text
