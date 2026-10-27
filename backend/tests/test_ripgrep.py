import json
import subprocess
from pathlib import Path

from app.search.ripgrep import run_ripgrep


def test_run_ripgrep_parses_windows_paths(monkeypatch, tmp_path: Path) -> None:
    file_path = tmp_path / "инструкция.txt"
    query = "Настройка служб IIS для работы по протоколу HTTPS"
    event = {
        "type": "match",
        "data": {
            "path": {"text": str(file_path)},
            "lines": {"text": f"{query}\n"},
            "line_number": 8,
        },
    }

    def fake_run(*args, **kwargs) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout=json.dumps(event),
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    hits = run_ripgrep(query, root=tmp_path)

    assert len(hits) == 1
    assert hits[0].file_path == file_path
    assert hits[0].rel_path == file_path.name
    assert hits[0].line_number == 8
    assert hits[0].snippet == query
