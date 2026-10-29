import json
import logging
import subprocess
from pathlib import Path
from typing import NamedTuple

logger = logging.getLogger(__name__)


class RipgrepHit(NamedTuple):
    file_path: Path
    rel_path: str
    line_number: int
    snippet: str
    context_before: list[str]
    context_after: list[str]


def run_ripgrep(
    query: str, root: Path | None = None, max_results: int | None = None
) -> list[RipgrepHit]:
    from app.config import settings

    root = root or settings.docs_root
    max_results = max_results or settings.ripgrep_max_results

    cmd = [
        "rg",
        "--json",
        "--fixed-strings",
        "-m",
        str(max_results),
        "--",
        query,
        str(root),
    ]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
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
        event = json.loads(line)
        if event["type"] != "match":
            continue

        data = event["data"]
        file_path = Path(data["path"]["text"])
        line_number = data["line_number"]
        snippet = data["lines"]["text"].strip()
        rel_path = str(file_path.relative_to(root))
        logger.debug(
            "run_ripgrep: parsed match | file_path=%r | line_number=%r | snippet=%r",
            file_path,
            line_number,
            snippet,
        )
        hits.append(
            RipgrepHit(
                file_path=file_path,
                rel_path=rel_path,
                line_number=line_number,
                snippet=snippet,
                context_before=[],
                context_after=[],
            )
        )

    return hits
