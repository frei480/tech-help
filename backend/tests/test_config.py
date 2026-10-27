import pytest

from app.config import Settings


def test_lm_studio_uses_lm_studio_credentials() -> None:
    config = Settings(
        _env_file=None,
        llm_provider="lm_studio",
        openai_api_base_url="http://lm-studio.example/v1",
        openai_api_key="lm-studio-key",
    )

    assert config.llm_provider == "lm_studio"
    assert config.llm_api_base_url == "http://lm-studio.example/v1"
    assert config.llm_api_key == "lm-studio-key"


def test_openrouter_uses_openrouter_credentials() -> None:
    config = Settings(
        _env_file=None,
        llm_provider="openrouter",
        openrouter_api_base_url="https://openrouter.example/v1",
        openrouter_api_key="test-key",
    )

    assert config.llm_api_base_url == "https://openrouter.example/v1"
    assert config.llm_api_key == "test-key"


def test_chat_client_uses_current_provider_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.llm.client as llm_client

    created: list[dict[str, object]] = []
    monkeypatch.setattr(
        llm_client,
        "ChatOpenAI",
        lambda **kwargs: created.append(kwargs) or kwargs,
    )
    llm_client._create_chat_model.cache_clear()

    monkeypatch.setattr(llm_client.settings, "llm_provider", "lm_studio")
    monkeypatch.setattr(llm_client.settings, "openai_api_base_url", "http://local.test/v1")
    monkeypatch.setattr(llm_client.settings, "openai_api_key", "local-key")
    monkeypatch.setattr(llm_client.settings, "chat_model", "local-model")
    llm_client.get_chat_model(temperature=0.3)

    monkeypatch.setattr(llm_client.settings, "llm_provider", "openrouter")
    monkeypatch.setattr(
        llm_client.settings,
        "openrouter_api_base_url",
        "https://router.test/api/v1",
    )
    monkeypatch.setattr(llm_client.settings, "openrouter_api_key", "router-key")
    monkeypatch.setattr(llm_client.settings, "chat_model", "router-model")
    llm_client.get_chat_model(temperature=0.3)

    assert created[0]["base_url"] == "http://local.test/v1"
    assert created[0]["model"] == "local-model"
    assert created[1]["base_url"] == "https://router.test/api/v1"
    assert created[1]["api_key"] == "router-key"
    assert created[1]["model"] == "router-model"
    llm_client._create_chat_model.cache_clear()
