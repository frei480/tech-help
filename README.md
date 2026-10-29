# Tech Help

Hybrid search MVP for local technical documentation (T-FLEX DOCs PLM/PDM).  
Combines **semantic (vector)** and **lexical (keyword)** search with **LLM-powered answer generation** via a LangGraph pipeline.

## Architecture

```
┌─────────────┐     ┌──────────────────────────────────────┐     ┌──────────┐
│  Frontend   │────▶│             Backend                  │────▶│   LLM    │
│  (React +   │     │  FastAPI + LangGraph + ChromaDB      │     │ Provider │
│   Vite)     │◀────│  + ripgrep                           │◀────│(OpenAI-  │
└─────────────┘     └──────────────────────────────────────┘     │compatible│
                                                                    └──────────┘
```

### Backend (`backend/`)

| Component | Description |
|---|---|
| [`app/main.py`](backend/app/main.py) | FastAPI app entry point |
| [`app/api/routes.py`](backend/app/api/routes.py) | REST & SSE endpoints: `/api/search`, `/api/search/stream`, `/api/ingest`, `/api/status`, `/api/files`, `/api/chunks` |
| [`app/api/llm_router.py`](backend/app/api/llm_router.py) | `/api/llm/models` and `/api/llm/embedding-models` — fetch chat and embedding models |
| [`app/graph.py`](backend/app/graph.py) | LangGraph state machine orchestrating the search pipeline |
| [`app/config.py`](backend/app/config.py) | Pydantic-based settings (`.env`-aware) |
| [`app/llm/prompts.py`](backend/app/llm/prompts.py) | System prompts for answer generation & reasoning |
| [`app/search/vector.py`](backend/app/search/vector.py) | Semantic search via ChromaDB |
| [`app/search/bm25.py`](backend/app/search/bm25.py) | BM25 ranking over the indexed documentation corpus |
| [`app/search/ripgrep.py`](backend/app/search/ripgrep.py) | Lexical search via ripgrep |
| [`app/search/fusion.py`](backend/app/search/fusion.py) | Reciprocal Rank Fusion of BM25, lexical, and vector results |
| [`app/search/query_expander.py`](backend/app/search/query_expander.py) | LLM-based query expansion with technical synonyms |
| [`app/search/keyword_extractor.py`](backend/app/search/keyword_extractor.py) | LLM-based keyword extraction from retrieved chunks |
| [`app/search/file_loader.py`](backend/app/search/file_loader.py) | Full-file loading tool (available to the answer LLM) |
| [`app/search/schemas.py`](backend/app/search/schemas.py) | Shared data schemas |
| [`app/ingestion/scanner.py`](backend/app/ingestion/scanner.py) | Scans `./docs` for `.txt`/`.md` files |
| [`app/ingestion/chunker.py`](backend/app/ingestion/chunker.py) | Token-aware text chunking (tiktoken `o200k_base`) |
| [`app/ingestion/chroma_store.py`](backend/app/ingestion/chroma_store.py) | ChromaDB persistence & embedding via OpenAI-compatible API |

### Frontend (`frontend/`)

| Component | Description |
|---|---|
| [`src/App.tsx`](frontend/src/App.tsx) | Root component |
| [`src/pages/Home.tsx`](frontend/src/pages/Home.tsx) | Main page: search bar, mode selector, model picker, results |
| [`src/api/client.ts`](frontend/src/api/client.ts) | API client: SSE streaming, REST calls |
| [`src/api/index.ts`](frontend/src/api/index.ts) | Re-exports |
| [`src/components/SearchResult.tsx`](frontend/src/components/SearchResult.tsx) | Displays answer + source chunks |
| [`src/components/ProcessTrace.tsx`](frontend/src/components/ProcessTrace.tsx) | Displays an answer and its supporting documentation chunks |
| [`src/styles.css`](frontend/src/styles.css) | Dark theme styles |

## Search Pipeline (LangGraph)

The search is a 9-node directed acyclic graph:

```
normalize → vector_search → extract_keywords_v → bm25_search → lexical_search_p1
    → extract_keywords_l → lexical_search_p2 → fuse → answer → END
```

1. **`normalize`** — trims the query
2. **`vector_search`** — semantic search via ChromaDB (embeddings from OpenAI-compatible API)
3. **`extract_keywords_v`** — LLM extracts key terms from vector hits
4. **`bm25_search`** — full-corpus BM25 search over indexed documentation chunks, using the original question and terms extracted from vector results
5. **`lexical_search_p1`** — ripgrep search using original query + LLM-expanded synonyms + vector keywords
6. **`extract_keywords_l`** — LLM extracts key terms from first lexical pass
7. **`lexical_search_p2`** — second ripgrep pass with newly discovered keywords
8. **`fuse`** — Reciprocal Rank Fusion of vector, BM25, and lexical results, deduplicated by source file
9. **`answer`** — LLM reviews the retrieved context, loading complete text for up to `FULL_FILE_MAX_FILES` distinct top-ranked source files. It checks each part of the question against the available evidence and gives a detailed answer covering relevant documented steps, conditions, and caveats, with citations. The corpus is local documentation only; the assistant states when the corpus cannot verify requested external or current information.

Internal model reasoning is not sent to or displayed by the frontend.

## Quick Start

### Prerequisites

- Docker & Docker Compose
- An OpenAI-compatible LLM provider (e.g. LM Studio, Ollama, OpenRouter)

### Setup

```bash
cp .env.example .env
# Edit .env to match your LLM provider
docker compose up --build
```

After changing backend code, rebuild the backend image to apply the changes:

```bash
docker compose up --build -d backend
```

### Endpoints

| Service | URL |
|---|---|
| Frontend | http://localhost:90 |
| Backend API | http://localhost:8001 |
| API docs (Swagger) | http://localhost:8001/docs |

### Usage

1. Open http://localhost:90
2. Click **Run ingestion** to index documentation files from `./docs/`
3. Enter a query and optionally select a specific model from the dropdown

## Configuration

All settings via `.env` file (see [`.env.example`](.env.example)). Set
`LLM_PROVIDER=lm_studio` to use LM Studio, or `LLM_PROVIDER=openrouter` to use OpenRouter.
For OpenRouter, set `OPENROUTER_API_KEY` and choose a `CHAT_MODEL` and
`EMBEDDING_MODEL` available from that provider. For LM Studio, set the model names
to models loaded in LM Studio.

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `lm_studio` | Provider: `lm_studio` or `openrouter` |
| `OPENAI_API_BASE_URL` | `http://host.docker.internal:1234/v1` | LM Studio API base URL |
| `OPENAI_API_KEY` | `lm-studio` | LM Studio API key (usually unused) |
| `OPENROUTER_API_BASE_URL` | `https://openrouter.ai/api/v1` | OpenRouter API base URL |
| `OPENROUTER_API_KEY` | — | OpenRouter API key (required for OpenRouter) |
| `CHAT_MODEL` | `qwen/qwen3.6-27b` | Model for answer generation |
| `EMBEDDING_MODEL` | `text-embedding-embeddinggemma-300m` | Model for embeddings |
| `DOCS_ROOT` | `./docs` | Documentation files directory |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | ChromaDB persistence directory |
| `FULL_FILE_MAX_FILES` | `3` | Maximum number of distinct top-ranked source files loaded in full for answer generation |
| `NPM_CONFIG_REGISTRY` | — | npm mirror URL (required for Docker build) |
| `PIP_INDEX_URL` | — | PyPI mirror URL (required for Docker build) |

## Development

### Backend

```bash
cd backend
pip install -e ".[dev]"
uvicorn app.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

### Tests

```bash
cd backend
pytest
```

## Project Structure

```
.
├── backend/
│   ├── app/
│   │   ├── api/           # FastAPI routes
│   │   ├── ingestion/     # Document scanning, chunking, ChromaDB
│   │   ├── llm/           # Prompts
│   │   └── search/        # Vector, lexical, fusion, expansion
│   ├── tests/
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── api/           # API client
│   │   ├── components/    # UI components
│   │   └── pages/         # Page components
│   └── Dockerfile
├── docs/                  # Documentation files (mounted read-only)
├── chroma_db/             # ChromaDB persistence (Docker volume)
├── docker-compose.yml
└── .env.example
