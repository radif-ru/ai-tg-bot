"""Тесты command-хендлеров (/start, /help и др.).

Стратегия: не поднимать реальный Dispatcher, вызывать функции хендлеров напрямую
с mock-объектом `Message`.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.handlers.commands import cmd_help, cmd_start
from app.services.model_registry import UserSettingsRegistry


@pytest.fixture
def fake_message() -> MagicMock:
    message = MagicMock()
    message.from_user.id = 42
    message.chat.id = 42
    message.answer = AsyncMock()
    return message


async def test_start_greets_user(fake_message: MagicMock) -> None:
    await cmd_start(fake_message)

    fake_message.answer.assert_awaited_once()
    text = fake_message.answer.call_args.args[0]
    assert "Привет" in text
    assert "/help" in text


async def test_help_contains_current_model(fake_message: MagicMock) -> None:
    registry = UserSettingsRegistry(
        default_model="qwen3.5:0.8b",
        default_prompt="Ты — полезный ассистент.",
    )

    await cmd_help(fake_message, registry)

    fake_message.answer.assert_awaited_once()
    text = fake_message.answer.call_args.args[0]
    assert "qwen3.5:0.8b" in text
    assert "/model" in text


async def test_help_truncates_long_prompt(fake_message: MagicMock) -> None:
    long_prompt = "A" * 500
    registry = UserSettingsRegistry(
        default_model="m",
        default_prompt=long_prompt,
    )

    await cmd_help(fake_message, registry)

    text = fake_message.answer.call_args.args[0]
    assert "AAA" in text
    # В тексте не должен быть весь длинный промпт — обрезается до 200 символов.
    assert "A" * 500 not in text
