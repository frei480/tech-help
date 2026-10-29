import logging
from typing import Literal

import httpx
from fastapi import APIRouter, HTTPException
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/llm", tags=["llm"])


class LlmSettingsUpdate(BaseModel):
    provider: Literal["lm_studio", "openrouter"]
    chat_model: str = Field(min_length=1)
    embedding_model: str = Field(min_length=1)


class LlmSettingsResponse(BaseModel):
    provider: Literal["lm_studio", "openrouter"]
    chat_model: str
    embedding_model: str


@router.get("/settings", response_model=LlmSettingsResponse)
async def get_settings() -> LlmSettingsResponse:
    return LlmSettingsResponse(
        provider=settings.llm_provider,
        chat_model=settings.chat_model,
        embedding_model=settings.embedding_model,
    )


@router.put("/settings", response_model=LlmSettingsResponse)
async def update_settings(request: LlmSettingsUpdate) -> LlmSettingsResponse:
    settings.llm_provider = request.provider
    settings.chat_model = request.chat_model
    settings.embedding_model = request.embedding_model
    from app.search.query_expander import expand_query_cached

    expand_query_cached.cache_clear()
    logger.info(
        "Updated LLM settings: provider=%s, chat_model=%s, embedding_model=%s",
        settings.llm_provider,
        settings.chat_model,
        settings.embedding_model,
    )
    return await get_settings()


@router.get("/models")
async def list_models(
    provider: Literal["lm_studio", "openrouter"] | None = None,
) -> list[dict[str, str]]:
    """Fetch available models from the configured OpenAI-compatible provider.

    Makes a GET /v1/models request to the provider and returns a simplified
    list of model IDs and names. Falls back to the configured chat_model
    if the provider is unreachable.
    """
    selected_provider = provider or settings.llm_provider
    if selected_provider == "openrouter":
        base_url = settings.openrouter_api_base_url
        api_key = settings.openrouter_api_key
    else:
        base_url = settings.openai_api_base_url
        api_key = settings.openai_api_key

    client = AsyncOpenAI(
        base_url=base_url,
        api_key=api_key,
        timeout=10.0,
        max_retries=0,
    )
    try:
        response = await client.models.list()
        models = [{"id": m.id, "name": m.id} for m in sorted(response.data, key=lambda x: x.id)]
        logger.info("Fetched %d models from %s", len(models), base_url)
        return models
    except Exception as exc:
        logger.warning(
            "Failed to fetch models from %s: %s. Falling back to configured model.",
            base_url,
            exc,
        )
        # Fallback: return the single configured model
        return [{"id": settings.chat_model, "name": settings.chat_model}]


@router.get("/embedding-models")
async def list_embedding_models(
    provider: Literal["lm_studio", "openrouter"] | None = None,
) -> list[dict[str, str]]:
    selected_provider = provider or settings.llm_provider
    if selected_provider == "lm_studio":
        return await list_models(provider="lm_studio")

    headers = {}
    if settings.openrouter_api_key:
        headers["Authorization"] = f"Bearer {settings.openrouter_api_key}"

    url = f"{settings.openrouter_api_base_url.rstrip('/')}/embeddings/models"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            payload = response.json()
    except httpx.HTTPError as exc:
        logger.warning("Failed to fetch OpenRouter embedding models from %s: %s", url, exc)
        raise HTTPException(
            status_code=502,
            detail=f"Failed to fetch OpenRouter embedding models: {exc}",
        ) from exc
    except ValueError as exc:
        logger.warning("Invalid JSON returned by OpenRouter embedding model endpoint: %s", exc)
        raise HTTPException(
            status_code=502,
            detail="OpenRouter returned an invalid embedding models response",
        ) from exc

    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        raise HTTPException(
            status_code=502,
            detail="OpenRouter returned an invalid embedding models response",
        )

    return [
        {"id": model["id"], "name": model.get("name") or model["id"]}
        for model in data
        if isinstance(model, dict) and isinstance(model.get("id"), str) and model["id"]
    ]
