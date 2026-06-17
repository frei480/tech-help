import logging

from fastapi import APIRouter, HTTPException
from openai import AsyncOpenAI

from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/llm", tags=["llm"])


@router.get("/models")
async def list_models() -> list[dict[str, str]]:
    """Fetch available models from the configured OpenAI-compatible provider.

    Makes a GET /v1/models request to the provider and returns a simplified
    list of model IDs and names. Falls back to the configured chat_model
    if the provider is unreachable.
    """
    client = AsyncOpenAI(
        base_url=settings.openai_api_base_url,
        api_key=settings.openai_api_key,
        timeout=10.0,
        max_retries=0,
    )
    try:
        response = await client.models.list()
        models = [{"id": m.id, "name": m.id} for m in sorted(response.data, key=lambda x: x.id)]
        logger.info("Fetched %d models from %s", len(models), settings.openai_api_base_url)
        return models
    except Exception as exc:
        logger.warning(
            "Failed to fetch models from %s: %s. Falling back to configured model.",
            settings.openai_api_base_url,
            exc,
        )
        # Fallback: return the single configured model
        return [{"id": settings.chat_model, "name": settings.chat_model}]
