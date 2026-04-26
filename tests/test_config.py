"""Тесты конфигурации (pydantic-settings)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError


def test_settings_loads_all_fields(base_env: dict[str, str]) -> None:
    from app.config import Settings

    settings = Settings(_env_file=None)

    assert settings.telegram_bot_token.get_secret_value() == base_env["TELEGRAM_BOT_TOKEN"]
    assert settings.ollama_base_url == base_env["OLLAMA_BASE_URL"]
    assert settings.ollama_default_model == base_env["OLLAMA_DEFAULT_MODEL"]
    assert settings.ollama_available_models == ["m1", "m2"]
    assert settings.ollama_timeout == 42
    assert settings.system_prompt == base_env["SYSTEM_PROMPT"]
    assert settings.log_level == base_env["LOG_LEVEL"]
    assert settings.log_file == base_env["LOG_FILE"]


def test_available_models_parsed_from_csv(
    monkeypatch: pytest.MonkeyPatch, base_env: dict[str, str]
) -> None:
    monkeypatch.setenv("OLLAMA_AVAILABLE_MODELS", "a, b ,c")
    monkeypatch.setenv("OLLAMA_DEFAULT_MODEL", "a")

    from app.config import Settings

    settings = Settings(_env_file=None)

    assert settings.ollama_available_models == ["a", "b", "c"]


def test_missing_token_raises(
    monkeypatch: pytest.MonkeyPatch, base_env: dict[str, str]
) -> None:
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN")

    from app.config import Settings

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_default_model_must_be_in_available(
    monkeypatch: pytest.MonkeyPatch, base_env: dict[str, str]
) -> None:
    monkeypatch.setenv("OLLAMA_DEFAULT_MODEL", "not-in-list")

    from app.config import Settings

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_secret_str_masks_token_in_repr(base_env: dict[str, str]) -> None:
    from app.config import Settings

    settings = Settings(_env_file=None)

    assert base_env["TELEGRAM_BOT_TOKEN"] not in repr(settings)
    assert base_env["TELEGRAM_BOT_TOKEN"] not in str(settings)


def test_history_defaults_are_sane(base_env: dict[str, str]) -> None:
    """Без переопределения env'ом должны действовать default'ы из Settings."""
    from app.config import Settings

    settings = Settings(_env_file=None)

    assert settings.history_max_messages == 20
    assert settings.history_summary_threshold == 10
    assert settings.history_summary_threshold <= settings.history_max_messages
    assert settings.summarization_prompt
    assert settings.log_llm_context is True


def test_history_env_overrides_apply(
    monkeypatch: pytest.MonkeyPatch, base_env: dict[str, str]
) -> None:
    monkeypatch.setenv("HISTORY_MAX_MESSAGES", "30")
    monkeypatch.setenv("HISTORY_SUMMARY_THRESHOLD", "5")
    monkeypatch.setenv("SUMMARIZATION_PROMPT", "Подведи итог.")
    monkeypatch.setenv("LOG_LLM_CONTEXT", "false")

    from app.config import Settings

    settings = Settings(_env_file=None)

    assert settings.history_max_messages == 30
    assert settings.history_summary_threshold == 5
    assert settings.summarization_prompt == "Подведи итог."
    assert settings.log_llm_context is False


def test_summary_threshold_above_max_raises(
    monkeypatch: pytest.MonkeyPatch, base_env: dict[str, str]
) -> None:
    monkeypatch.setenv("HISTORY_MAX_MESSAGES", "10")
    monkeypatch.setenv("HISTORY_SUMMARY_THRESHOLD", "20")

    from app.config import Settings

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_history_max_zero_raises(
    monkeypatch: pytest.MonkeyPatch, base_env: dict[str, str]
) -> None:
    monkeypatch.setenv("HISTORY_MAX_MESSAGES", "0")

    from app.config import Settings

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_summary_threshold_zero_raises(
    monkeypatch: pytest.MonkeyPatch, base_env: dict[str, str]
) -> None:
    monkeypatch.setenv("HISTORY_SUMMARY_THRESHOLD", "0")

    from app.config import Settings

    with pytest.raises(ValidationError):
        Settings(_env_file=None)
