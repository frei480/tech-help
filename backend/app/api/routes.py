import json
import logging
import sys
from typing import Any, AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.config import settings
from app.graph import app as graph_app
from app.ingestion.chroma_store import ChromaStore
from app.ingestion.chunker import chunk_text
from app.ingestion.scanner import scan_docs
from app.api.llm_router import router as llm_router

# Гарантируем, что логи INFO попадают в stdout (видно в docker logs)
# Используем force=True, чтобы переопределить логгер, уже настроенный uvicorn
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
    force=True,
)
# Дополнительно настраиваем корневой логгер и логгер uvicorn, чтобы логи точно доходили
for logger_name in ("uvicorn", "uvicorn.access", "uvicorn.error", ""):
    log = logging.getLogger(logger_name)
    log.setLevel(logging.INFO)
    if not log.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        log.addHandler(handler)

app = FastAPI(title="Tech Help Search")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(llm_router)


class SearchRequest(BaseModel):
    query: str
    mode: str = "ai_answer"
    model: str | None = None


class IngestResponse(BaseModel):
    indexed_files: int
    indexed_chunks: int


class StatusResponse(BaseModel):
    indexed_files: int
    indexed_chunks: int


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"


logger = logging.getLogger(__name__)


@app.get("/api/status", response_model=StatusResponse)
async def status() -> StatusResponse:
    store = ChromaStore()
    files = len(
        {m.get("rel_path") for m in store.collection.get(include=["metadatas"])["metadatas"]}
    )
    return StatusResponse(indexed_files=files, indexed_chunks=store.count())


@app.get("/api/files")
async def list_files() -> list[dict]:
    store = ChromaStore()
    results = store.collection.get(include=["metadatas"])
    files: dict[str, dict] = {}
    for meta in results["metadatas"]:
        rel = meta.get("rel_path", "unknown")
        files[rel] = {
            "rel_path": rel,
            "file_path": meta.get("file_path", rel),
            "chunks": sum(1 for m in results["metadatas"] if m.get("rel_path") == rel),
        }
    return list(files.values())


@app.get("/api/chunks")
async def get_chunks(file: str) -> list[dict]:
    store = ChromaStore()
    results = store.collection.get(where={"rel_path": file}, include=["documents", "metadatas"])
    return [{"text": d, "metadata": m} for d, m in zip(results["documents"], results["metadatas"])]


@app.post("/api/ingest", response_model=IngestResponse)
async def ingest() -> IngestResponse:
    files = scan_docs()
    store = ChromaStore()
    total_chunks = 0
    for file_path in files:
        rel = str(file_path.relative_to(settings.docs_root))
        store.delete_by_file(rel)
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        total_chunks += store.index_chunks(chunk_text(text, file_path, rel))
    return IngestResponse(indexed_files=len(files), indexed_chunks=total_chunks)


@app.post("/api/search")
async def search(request: SearchRequest) -> dict:
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty")

    initial_state = {
        "query": request.query,
        "mode": request.mode,
        "model": request.model,
        "lexical_hits": [],
        "vector_hits": [],
        "fused": [],
    }
    final_state = await graph_app.ainvoke(initial_state)

    return {
        "query": request.query,
        "mode": request.mode,
        "answer": final_state.get("answer"),
        "chunks": [
            {"text": i.get("text"), "metadata": i.get("metadata")}
            for i in final_state.get("fused", [])
        ],
        "trace": [],
    }


@app.post("/api/search/stream")
async def search_stream(request: SearchRequest) -> StreamingResponse:
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty")

    async def generate_events() -> AsyncGenerator[str, None]:
        try:
            initial_state: dict[str, Any] = {
                "query": request.query.strip(),
                "mode": request.mode,
                "model": request.model,
            }

            final_state: dict[str, Any] = {}

            # We use stream_mode=["custom", "updates"] because the generate_answer
            # node uses the OpenAI client directly (not LangChain's ChatOpenAI) to
            # properly capture vendor-specific delta fields like reasoning_content.
            #
            # The generate_answer node emits custom events via StreamWriter:
            #   ("token", "some text")       — regular content tokens
            #   ("reasoning_token", "some")  — reasoning content tokens
            #
            # All other nodes now also emit custom events:
            #   ("node_start", {...})   — when a node begins execution
            #   ("node_result", {...})  — when a node completes with its results
            #
            # These are received here as event_type="custom" events and forwarded
            # to the client as SSE events in real-time.
            #
            # The final state (with complete answer, reasoning_content, and chunks)
            # is received via event_type="updates" and sent as a "done" event.
            async for event_type, payload in graph_app.astream(
                initial_state,
                stream_mode=["custom", "updates"],
            ):
                if event_type == "custom":
                    # Custom events from StreamWriter — forward as SSE
                    event_name, data = payload
                    if event_name == "token":
                        yield _sse("token", {"token": data})
                    elif event_name == "reasoning_token":
                        yield _sse("reasoning_token", {"token": data})
                    elif event_name == "node_start":
                        yield _sse("node_start", data)
                    elif event_name == "node_result":
                        yield _sse("node_result", data)
                elif event_type == "updates":
                    for node_update in payload.values():
                        final_state.update(node_update)

            # After the loop, send the done event with complete data
            if "answer" in final_state:
                yield _sse(
                    "done",
                    {
                        "query": request.query,
                        "mode": request.mode,
                        "answer": final_state.get("answer"),
                        "reasoning_content": final_state.get("reasoning_content"),
                        "chunks": [
                            {
                                "text": i.get("text"),
                                "metadata": i.get("metadata"),
                                "score": i.get("score"),
                            }
                            for i in final_state.get("fused", [])
                        ],
                    },
                )
            else:
                logger.error("[stream] No final state captured from graph")
                yield _sse("error", {"message": "Graph did not produce a final state"})

        except GeneratorExit:
            logger.warning("[stream] Client disconnected mid-stream")
            return
        except Exception as exc:
            logger.exception("[stream] Error in generate_events")
            yield _sse("error", {"message": str(exc)})

    return StreamingResponse(generate_events(), media_type="text/event-stream")
