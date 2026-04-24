"""Общие фикстуры для тестов."""

from __future__ import annotations

from collections.abc import Iterator

import pytest


@pytest.fixture
def base_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, str]]:
    """Базовый валидный набор переменных окружения для Settings.

    Тесты могут переопределять отдельные значения через свой monkeypatch.
    """
    env = {
        "TELEGRAM_BOT_TOKEN": "123:abc",
        "OLLAMA_BASE_URL": "http://localhost:11434",
        "OLLAMA_DEFAULT_MODEL": "m1",
        "OLLAMA_AVAILABLE_MODELS": "m1,m2",
        "OLLAMA_TIMEOUT": "42",
        "SYSTEM_PROMPT": "You are helpful.",
        "LOG_LEVEL": "DEBUG",
        "LOG_FILE": "logs/test.log",
    }
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    yield env
