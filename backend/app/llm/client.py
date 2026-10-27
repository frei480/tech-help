from functools import lru_cache

from langchain_openai import ChatOpenAI

from app.config import settings


@lru_cache(maxsize=8)
def _create_chat_model(
    base_url: str,
    api_key: str,
    model: str,
    temperature: float,
    timeout: float,
) -> ChatOpenAI:
    return ChatOpenAI(
        base_url=base_url,
        api_key=api_key,
        model=model,
        temperature=temperature,
        timeout=timeout,
    )


def get_chat_model(*, temperature: float, timeout: float = 30.0) -> ChatOpenAI:
    return _create_chat_model(
        base_url=settings.llm_api_base_url,
        api_key=settings.llm_api_key,
        model=settings.chat_model,
        temperature=temperature,
        timeout=timeout,
    )
