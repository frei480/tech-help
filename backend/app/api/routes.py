import json
import logging
import sys
from typing import Any, AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI
from pydantic import BaseModel

from app.config import settings
from app.graph import app as graph_app, lexical_search_tool, vector_search_tool
from app.ingestion.chroma_store import ChromaStore
from app.ingestion.chunker import chunk_text
from app.ingestion.scanner import scan_docs
from app.llm.prompts import build_prompt
from app.search.fusion import weighted_fusion
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

# OpenAI-клиент для стриминга напрямую (не через LangChain),
# чтобы гарантированно получать токены от любых OpenAI-совместимых провайдеров
_stream_client = AsyncOpenAI(
    base_url=settings.openai_api_base_url,
    api_key=settings.openai_api_key,
    timeout=300.0,
    max_retries=0,
)

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


def _extract_token_text(chunk: Any) -> str | None:
    """Извлекает текст токена из AIMessageChunk, учитывая различные форматы."""
    # Прямой доступ к .content (строка)
    if hasattr(chunk, "content"):
        content = chunk.content
        if isinstance(content, str) and content:
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    parts.append(item["text"])
            if parts:
                return "".join(parts)
    # Некоторые провайдеры кладут текст в additional_kwargs
    if hasattr(chunk, "additional_kwargs") and chunk.additional_kwargs:
        delta = chunk.additional_kwargs
        if isinstance(delta, dict):
            content_str = delta.get("content", "")
            if content_str:
                return str(content_str)
    # Некоторые провайдеры (например, LM Studio) кладут контент в tool_call_chunks
    if hasattr(chunk, "tool_call_chunks") and chunk.tool_call_chunks:
        for tcc in chunk.tool_call_chunks:
            if hasattr(tcc, "text") and tcc.text:
                return tcc.text
            if hasattr(tcc, "content") and tcc.content:
                return tcc.content
    # Попробуем найти текст в usage_metadata
    if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
        logger.info("chunk.usage_metadata=%r", chunk.usage_metadata)
    # Полная диагностика: логируем все публичные атрибуты chunk
    public_attrs = {k: repr(v) for k, v in chunk.__dict__.items() if not k.startswith("_")}
    # Также пытаемся получить сырой JSON через дополнительный канал
    logger.warning(
        "Failed to extract token from chunk. type=%s content=%r additional_kwargs=%r "
        "response_metadata=%r id=%r public_attrs=%r",
        type(chunk).__name__,
        getattr(chunk, "content", "N/A"),
        getattr(chunk, "additional_kwargs", "N/A"),
        getattr(chunk, "response_metadata", "N/A"),
        getattr(chunk, "id", "N/A"),
        public_attrs,
    )
    return None


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
        "reasoning_summary": final_state.get("reasoning_summary", []),
        "source_notes": final_state.get("source_notes", []),
        "limitations": final_state.get("limitations", []),
    }


@app.post("/api/search/stream")
async def search_stream(request: SearchRequest) -> StreamingResponse:
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty")

    async def generate_events() -> AsyncGenerator[str, None]:
        try:
            # ── 1. Поиск через инструменты ──────────────────────────────────
            query = request.query.strip()
            logger.info("[stream] Starting generate_events for query=%s", query)
            lexical_result = lexical_search_tool.invoke({"query": query})
            vector_result = vector_search_tool.invoke({"query": query})

            lexical_hits = lexical_result if isinstance(lexical_result, list) else []
            vector_hits = vector_result if isinstance(vector_result, list) else []

            # ── 2. Фуззинг ───────────────────────────────────────────────────
            fused_items = weighted_fusion(
                lexical_hits=lexical_hits,
                vector_hits=vector_hits,
                top_k=settings.final_top_k,
            )
            fused = [item["item"] for item in fused_items]

            answer: str | None = None
            reasoning_summary: list[str] = []
            source_notes: list[dict[str, Any]] = []
            limitations: list[str] = []

            logger.info(
                "[stream] Stream state: mode=%s fused_count=%d query=%s",
                request.mode,
                len(fused),
                query,
            )

            # ── 3. Стриминг ответа LLM через openai SDK напрямую ────────────
            if request.mode == "ai_answer" and fused:
                model_name = request.model or settings.chat_model
                prompt = build_prompt(query, fused, include_reasoning=False)
                logger.info(
                    "[stream] Starting LLM stream via openai SDK, model=%s prompt_length=%d",
                    model_name,
                    len(prompt),
                )
                try:
                    chunk_count = 0
                    first_chunk = True
                    stream = await _stream_client.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "user", "content": prompt}],
                        stream=True,
                    )
                    async for chunk in stream:
                        chunk_count += 1
                        delta = chunk.choices[0].delta if chunk.choices else None
                        if delta:
                            # reasoning_content — отправляем отдельным событием
                            if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                                yield _sse("reasoning_token", {"token": delta.reasoning_content})
                            # content — основной ответ, накапливаем в answer
                            if delta.content:
                                yield _sse("token", {"token": delta.content})
                                if answer is None:
                                    answer = delta.content
                                else:
                                    answer += delta.content
                        if first_chunk:
                            logger.info(
                                "[stream] FIRST chunk #%d: delta=%r reasoning=%r content=%r",
                                chunk_count,
                                delta,
                                getattr(delta, "reasoning_content", None),
                                delta.content if delta else None,
                            )
                            first_chunk = False
                    logger.info(
                        "[stream] LLM stream finished, total chunks=%d answer_length=%d",
                        chunk_count,
                        len(answer or ""),
                    )
                    if chunk_count == 0:
                        logger.warning(
                            "[stream] LLM stream produced ZERO chunks — possible streaming issue"
                        )
                except Exception as e:
                    logger.exception("[stream] LLM stream error: %s", e)
                    yield _sse("error", {"message": f"LLM stream error: {e}"})
                    return

                # ── 4. Генерация reasoning (через openai SDK, без стриминга) ─
                reasoning_prompt = build_prompt(query, fused, include_reasoning=True)
                messages = [
                    {"role": "user", "content": answer or ""},
                    {"role": "user", "content": reasoning_prompt},
                ]
                try:
                    result = await _stream_client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        stream=False,
                    )
                    text = result.choices[0].message.content or ""
                    start = text.find("{")
                    end = text.rfind("}")
                    if start != -1 and end != -1:
                        parsed = json.loads(text[start : end + 1])
                        reasoning_summary = parsed.get("reasoning_summary", [])
                        source_notes = parsed.get("source_notes", [])
                        limitations = parsed.get("limitations", [])
                except Exception:
                    pass

            logger.info("[stream] Yielding done event for query=%s", query)
            yield _sse(
                "done",
                {
                    "query": request.query,
                    "mode": request.mode,
                    "answer": answer,
                    "chunks": [
                        {
                            "text": i.get("text"),
                            "metadata": i.get("metadata"),
                            "score": i.get("score"),
                        }
                        for i in fused
                    ],
                    "reasoning_summary": reasoning_summary,
                    "source_notes": source_notes,
                    "limitations": limitations,
                },
            )

        except GeneratorExit:
            logger.warning(
                "[stream] GeneratorExit caught for query=%s — client disconnected mid-stream",
                request.query,
            )
            return
        except Exception as exc:
            logger.exception(
                "[stream] Unhandled exception in generate_events for query=%s", request.query
            )
            yield _sse("error", {"message": str(exc)})

    return StreamingResponse(generate_events(), media_type="text/event-stream")
