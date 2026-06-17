import pytest
from fastapi.testclient import TestClient

from app.api.routes import app

client = TestClient(app)


def test_ingest_missing_root() -> None:
    response = client.post("/api/ingest")
    assert response.status_code == 200
    payload = response.json()
    assert "indexed_files" in payload
