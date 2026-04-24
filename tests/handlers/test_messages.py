"""Тесты обработчика произвольного текста → LLM."""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.handlers.messages import MAX_INPUT_LENGTH, handle_text
from app.services.llm import LLMBadResponse, LLMError, LLMTimeout, LLMUnavailable
from app.services.model_registry import UserSettingsRegistry


def _make_message(text: str = "hello", user_id: int = 42, chat_id: int = 42) -> MagicMock:
    message = MagicMock()
    message.from_user.id = user_id
    message.chat.id = chat_id
    message.text = text
    message.answer = AsyncMock()
    message.bot = MagicMock()
    message.bot.send_chat_action = AsyncMock()
    return message


def _make_registry() -> UserSettingsRegistry:
    return UserSettingsRegistry(
        default_model="qwen3.5:0.8b",
        default_prompt="Ты — полезный ассистент.",
    )


async def test_success_path_returns_llm_response() -> None:
    registry = _make_registry()
    llm_client = MagicMock()
    llm_client.generate = AsyncMock(return_value="pong")
    message = _make_message(text="ping")

    await handle_text(message, llm_client, registry)

    llm_client.generate.assert_awaited_once_with(
        "ping", model="qwen3.5:0.8b", system_prompt="Ты — полезный ассистент."
    )
    message.bot.send_chat_action.assert_awaited()
    message.answer.assert_awaited_once_with("pong")


async def test_timeout_answers_about_slowness_and_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    registry = _make_registry()
    llm_client = MagicMock()
    llm_client.generate = AsyncMock(side_effect=LLMTimeout("t"))
    message = _make_message(text="ping")

    caplog.set_level(logging.WARNING, logger="app.handlers.messages")

    await handle_text(message, llm_client, registry)

    text = message.answer.call_args.args[0]
    assert "долго" in text
    assert any(record.levelname == "WARNING" for record in caplog.records)


async def test_unavailable_answers_about_downtime_and_errors(
    caplog: pytest.LogCaptureFixture,
) -> None:
    registry = _make_registry()
    llm_client = MagicMock()
    llm_client.generate = AsyncMock(side_effect=LLMUnavailable("down"))
    message = _make_message(text="ping")

    caplog.set_level(logging.ERROR, logger="app.handlers.messages")

    await handle_text(message, llm_client, registry)

    text = message.answer.call_args.args[0]
    assert "недоступна" in text
    assert any(record.levelname == "ERROR" for record in caplog.records)


async def test_bad_response_uses_exception_message_for_model_not_found() -> None:
    registry = _make_registry()
    llm_client = MagicMock()
    llm_client.generate = AsyncMock(side_effect=LLMBadResponse("модель не найдена"))
    message = _make_message(text="ping")

    await handle_text(message, llm_client, registry)

    text = message.answer.call_args.args[0]
    assert "/models" in text or "не найдена" in text


async def test_generic_llm_error_returns_generic_message() -> None:
    registry = _make_registry()
    llm_client = MagicMock()
    llm_client.generate = AsyncMock(side_effect=LLMError("unknown"))
    message = _make_message(text="ping")

    await handle_text(message, llm_client, registry)

    text = message.answer.call_args.args[0]
    assert "ошибка" in text.lower()


async def test_long_response_is_split_into_chunks() -> None:
    registry = _make_registry()
    llm_client = MagicMock()
    long_response = "x" * 10000  # > 4096
    llm_client.generate = AsyncMock(return_value=long_response)
    message = _make_message(text="ping")

    await handle_text(message, llm_client, registry)

    assert message.answer.await_count >= 3


async def test_too_long_input_is_rejected_without_llm_call() -> None:
    registry = _make_registry()
    llm_client = MagicMock()
    llm_client.generate = AsyncMock()
    too_long = "x" * (MAX_INPUT_LENGTH + 1)
    message = _make_message(text=too_long)

    await handle_text(message, llm_client, registry)

    llm_client.generate.assert_not_awaited()
    text = message.answer.call_args.args[0]
    assert "сократите" in text.lower() or "слишком" in text.lower()


async def test_handler_logs_user_chat_model_dur_ms(
    caplog: pytest.LogCaptureFixture,
) -> None:
    registry = _make_registry()
    registry.set_model(42, "deepseek-r1:1.5b")
    llm_client = MagicMock()
    llm_client.generate = AsyncMock(return_value="ok")
    message = _make_message(text="hi", user_id=42, chat_id=77)

    caplog.set_level(logging.INFO, logger="app.handlers.messages")

    await handle_text(message, llm_client, registry)

    assert any(
        "user=42" in rec.message
        and "chat=77" in rec.message
        and "model=deepseek-r1:1.5b" in rec.message
        and "dur_ms=" in rec.message
        for rec in caplog.records
    )
