"""Тесты command-хендлеров (/start, /help, /models, /model, /prompt).

Стратегия: не поднимать реальный Dispatcher, вызывать функции хендлеров напрямую
с mock-объектом `Message`.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiogram.filters import CommandObject

from app.handlers.commands import (
    cmd_help,
    cmd_model,
    cmd_models,
    cmd_prompt,
    cmd_start,
)
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


def _fake_settings(models: list[str] | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        ollama_available_models=models or ["qwen3.5:0.8b", "deepseek-r1:1.5b"]
    )


def _cmd(command: str, args: str | None = None) -> CommandObject:
    return CommandObject(prefix="/", command=command, args=args)


async def test_models_lists_all_and_marks_active(fake_message: MagicMock) -> None:
    settings = _fake_settings(["qwen3.5:0.8b", "deepseek-r1:1.5b"])
    registry = UserSettingsRegistry(
        default_model="deepseek-r1:1.5b",
        default_prompt="p",
    )

    await cmd_models(fake_message, settings, registry)

    text = fake_message.answer.call_args.args[0]
    assert "qwen3.5:0.8b" in text
    assert "deepseek-r1:1.5b" in text
    assert "активная" in text


async def test_model_switches_to_valid_name(fake_message: MagicMock) -> None:
    settings = _fake_settings()
    registry = MagicMock(spec=UserSettingsRegistry)

    await cmd_model(
        fake_message, _cmd("model", "qwen3.5:0.8b"), settings, registry
    )

    registry.set_model.assert_called_once_with(42, "qwen3.5:0.8b")
    text = fake_message.answer.call_args.args[0]
    assert "переключена" in text


async def test_model_rejects_unknown_name(fake_message: MagicMock) -> None:
    settings = _fake_settings()
    registry = MagicMock(spec=UserSettingsRegistry)

    await cmd_model(fake_message, _cmd("model", "unknown-model"), settings, registry)

    registry.set_model.assert_not_called()
    text = fake_message.answer.call_args.args[0]
    assert "не найдена" in text


async def test_model_without_arg_shows_usage(fake_message: MagicMock) -> None:
    settings = _fake_settings()
    registry = MagicMock(spec=UserSettingsRegistry)

    await cmd_model(fake_message, _cmd("model", None), settings, registry)

    registry.set_model.assert_not_called()
    text = fake_message.answer.call_args.args[0]
    assert "/model" in text and "/models" in text


async def test_prompt_sets_custom_text(fake_message: MagicMock) -> None:
    registry = MagicMock(spec=UserSettingsRegistry)

    await cmd_prompt(fake_message, _cmd("prompt", "Отвечай как пират"), registry)

    registry.set_prompt.assert_called_once_with(42, "Отвечай как пират")
    registry.reset_prompt.assert_not_called()
    text = fake_message.answer.call_args.args[0]
    assert "обновлён" in text


async def test_prompt_without_arg_resets(fake_message: MagicMock) -> None:
    registry = MagicMock(spec=UserSettingsRegistry)

    await cmd_prompt(fake_message, _cmd("prompt", None), registry)

    registry.reset_prompt.assert_called_once_with(42)
    registry.set_prompt.assert_not_called()
    text = fake_message.answer.call_args.args[0]
    assert "сброшен" in text


async def test_model_whitespace_only_arg_treated_as_empty(
    fake_message: MagicMock,
) -> None:
    settings = _fake_settings()
    registry = MagicMock(spec=UserSettingsRegistry)

    await cmd_model(fake_message, _cmd("model", "   "), settings, registry)

    registry.set_model.assert_not_called()
    text = fake_message.answer.call_args.args[0]
    assert "/model" in text
