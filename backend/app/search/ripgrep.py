import subprocess
from pathlib import Path
from typing import NamedTuple


class RipgrepHit(NamedTuple):
    file_path: Path
    rel_path: str
    line_number: int
    snippet: str
    context_before: list[str]
    context_after: list[str]


def run_ripgrep(query: str, root: Path | None = None, max_results: int | None = None) -> list[RipgrepHit]:
    from app.config import settings

    root = root or settings.docs_root
    max_results = max_results or settings.ripgrep_max_results

    cmd = [
        "rg",
        "--line-number",
        "--no-heading",
        "--no-filename",
        "--context",
        "2",
        "--max-columns",
        "300",
        "-m",
        str(max_results),
        query,
        str(root),
    ]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "ripgrep binary not found. Install ripgrep or disable lexical search."
        ) from exc

    hits: list[RipgrepHit] = []
    if proc.returncode not in (0, 1):
        raise RuntimeError(f"rg failed: {proc.stderr}")

    for line in proc.stdout.splitlines():
        parts = line.split(":", 2)
        if len(parts) < 3:
            continue
        file_str, line_no_str, content = parts
        hits.append(
            RipgrepHit(
                file_path=Path(file_str),
                rel_path=str(Path(file_str).relative_to(root)),
                line_number=int(line_no_str),
                snippet=content.strip(),
                context_before=[],
                context_after=[],
            )
        )

    return hits
