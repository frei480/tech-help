import os
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


DEFAULT_TIKTOKEN_CACHE_DIR = Path(__file__).resolve().parents[1] / "tiktoken_cache"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm_provider: Literal["lm_studio", "openrouter"] = "lm_studio"
    openai_api_base_url: str = "http://localhost:1234/v1"
    openai_api_key: str = "lm-studio"
    openrouter_api_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_api_key: str = ""
    chat_model: str = "qwen/qwen3.6-27b"
    embedding_model: str = "text-embedding-embeddinggemma-300m"
    tiktoken_cache_dir: Path = DEFAULT_TIKTOKEN_CACHE_DIR

    docs_root: Path = Path("./docs")
    chroma_persist_dir: Path = Path("./chroma_db")
    chunk_size_tokens: int = 1000
    chunk_overlap_tokens: int = 150

    ripgrep_max_results: int = 50
    fusion_top_k: int = 10
    final_top_k: int = 5

    full_file_max_files: int = 3
    full_file_max_tokens: int = 8_000

    @property
    def chroma_collection_name(self) -> str:
        return "docs_chunks"

    @property
    def llm_api_base_url(self) -> str:
        if self.llm_provider == "openrouter":
            return self.openrouter_api_base_url
        return self.openai_api_base_url

    @property
    def llm_api_key(self) -> str:
        if self.llm_provider == "openrouter":
            return self.openrouter_api_key
        return self.openai_api_key


settings = Settings()
os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(settings.tiktoken_cache_dir))
