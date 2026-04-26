"""Тесты обработчика произвольного текста → LLM с историей и суммаризацией."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.handlers.messages import MAX_INPUT_LENGTH, handle_text
from app.services.conversation import SUMMARY_PREFIX, ConversationStore
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


def _make_settings(
    *,
    log_llm_context: bool = True,
    history_max_messages: int = 20,
    history_summary_threshold: int = 100,  # по умолчанию суммаризация не сработает
) -> SimpleNamespace:
    """Лёгкий заменитель Settings для тестов handler'а.

    Реальный `Settings` требует валидный TELEGRAM_BOT_TOKEN и т.п.; handler
    использует ровно три поля, поэтому мокаем их через SimpleNamespace.
    """
    return SimpleNamespace(
        log_llm_context=log_llm_context,
        history_max_messages=history_max_messages,
        history_summary_threshold=history_summary_threshold,
    )


def _make_summarizer(mocker, *, summary_text: str = "резюме") -> MagicMock:
    s = MagicMock()
    s.summarize = AsyncMock(return_value=summary_text)
    return s


# --- успешный путь + контекст ---


async def test_success_path_sends_system_plus_history_to_chat() -> None:
    registry = _make_registry()
    llm_client = MagicMock()
    llm_client.chat = AsyncMock(return_value="pong")
    conversation = ConversationStore(max_messages=20)
    summarizer = MagicMock()
    summarizer.summarize = AsyncMock()
    settings = _make_settings()
    message = _make_message(text="ping")

    await handle_text(
        message, llm_client, registry, conversation, summarizer, settings
    )

    llm_client.chat.assert_awaited_once()
    call_args = llm_client.chat.await_args
    payload = call_args.args[0]
    assert call_args.kwargs == {"model": "qwen3.5:0.8b"}
    assert payload[0] == {"role": "system", "content": "Ты — полезный ассистент."}
    assert payload[1:] == [{"role": "user", "content": "ping"}]
    message.bot.send_chat_action.assert_awaited()
    message.answer.assert_awaited_once_with("pong")


async def test_history_is_updated_after_successful_response() -> None:
    registry = _make_registry()
    llm_client = MagicMock()
    llm_client.chat = AsyncMock(return_value="pong")
    conversation = ConversationStore(max_messages=20)
    summarizer = _make_summarizer(MagicMock())
    settings = _make_settings()
    message = _make_message(text="ping", user_id=7)

    await handle_text(
        message, llm_client, registry, conversation, summarizer, settings
    )

    assert conversation.get_history(user_id=7) == [
        {"role": "user", "content": "ping"},
        {"role": "assistant", "content": "pong"},
    ]


async def test_second_message_includes_first_pair_in_context() -> None:
    """Проверяем, что бот «помнит» предыдущий обмен."""
    registry = _make_registry()
    llm_client = MagicMock()
    llm_client.chat = AsyncMock(side_effect=["первый ответ", "второй ответ"])
    conversation = ConversationStore(max_messages=20)
    summarizer = _make_summarizer(MagicMock())
    settings = _make_settings()

    msg1 = _make_message(text="первое", user_id=7)
    await handle_text(msg1, llm_client, registry, conversation, summarizer, settings)
    msg2 = _make_message(text="второе", user_id=7)
    await handle_text(msg2, llm_client, registry, conversation, summarizer, settings)

    second_call_payload = llm_client.chat.await_args_list[1].args[0]
    # [system, user1, assistant1, user2]
    assert len(second_call_payload) == 4
    assert second_call_payload[0]["role"] == "system"
    assert second_call_payload[1] == {"role": "user", "content": "первое"}
    assert second_call_payload[2] == {"role": "assistant", "content": "первый ответ"}
    assert second_call_payload[3] == {"role": "user", "content": "второе"}


# --- логирование контекста (требование ТЗ §5) ---


async def test_context_log_with_payload_when_log_llm_context_true(
    caplog: pytest.LogCaptureFixture,
) -> None:
    registry = _make_registry()
    llm_client = MagicMock()
    llm_client.chat = AsyncMock(return_value="ok")
    conversation = ConversationStore(max_messages=20)
    summarizer = _make_summarizer(MagicMock())
    settings = _make_settings(log_llm_context=True)
    message = _make_message(text="hi", user_id=42, chat_id=77)

    caplog.set_level(logging.INFO, logger="app.handlers.messages")

    await handle_text(
        message, llm_client, registry, conversation, summarizer, settings
    )

    context_lines = [
        rec.message for rec in caplog.records if "llm_context" in rec.message
    ]
    assert context_lines, "ожидали строку llm_context в логе"
    line = context_lines[0]
    assert "user=42" in line
    assert "chat=77" in line
    assert "model=qwen3.5:0.8b" in line
    assert "messages=2" in line  # system + user
    assert "tokens=" in line
    assert "payload=" in line


async def test_context_log_without_payload_when_log_llm_context_false(
    caplog: pytest.LogCaptureFixture,
) -> None:
    registry = _make_registry()
    llm_client = MagicMock()
    llm_client.chat = AsyncMock(return_value="ok")
    conversation = ConversationStore(max_messages=20)
    summarizer = _make_summarizer(MagicMock())
    settings = _make_settings(log_llm_context=False)
    message = _make_message(text="hi")

    caplog.set_level(logging.INFO, logger="app.handlers.messages")

    await handle_text(
        message, llm_client, registry, conversation, summarizer, settings
    )

    context_lines = [
        rec.message for rec in caplog.records if "llm_context" in rec.message
    ]
    assert context_lines
    line = context_lines[0]
    assert "messages=" in line
    assert "tokens=" in line
    assert "payload=" not in line


# --- суммаризация ---


async def test_summarizer_called_when_history_reaches_threshold() -> None:
    registry = _make_registry()
    llm_client = MagicMock()
    llm_client.chat = AsyncMock(return_value="pong")
    conversation = ConversationStore(max_messages=20)
    summarizer = MagicMock()
    summarizer.summarize = AsyncMock(return_value="итог")
    # Порог=4 → после первого ответа история = [user, assistant] (2)
    # → после второго ответа история = [user, assistant, user, assistant] (4) → суммаризация.
    settings = _make_settings(history_summary_threshold=4)

    msg1 = _make_message(text="m1", user_id=9)
    await handle_text(msg1, llm_client, registry, conversation, summarizer, settings)
    msg2 = _make_message(text="m2", user_id=9)
    await handle_text(msg2, llm_client, registry, conversation, summarizer, settings)

    summarizer.summarize.assert_awaited_once()
    history = conversation.get_history(user_id=9)
    # После replace_with_summary: [system: SUMMARY_PREFIX+итог, user2, assistant2].
    assert len(history) == 3
    assert history[0]["role"] == "system"
    assert history[0]["content"] == f"{SUMMARY_PREFIX}итог"
    assert history[1] == {"role": "user", "content": "m2"}
    assert history[2] == {"role": "assistant", "content": "pong"}


async def test_summarizer_not_called_below_threshold() -> None:
    registry = _make_registry()
    llm_client = MagicMock()
    llm_client.chat = AsyncMock(return_value="pong")
    conversation = ConversationStore(max_messages=20)
    summarizer = MagicMock()
    summarizer.summarize = AsyncMock()
    settings = _make_settings(history_summary_threshold=10)
    message = _make_message(text="ping")

    await handle_text(
        message, llm_client, registry, conversation, summarizer, settings
    )

    summarizer.summarize.assert_not_awaited()


async def test_summarizer_failure_does_not_break_response(
    caplog: pytest.LogCaptureFixture,
) -> None:
    registry = _make_registry()
    llm_client = MagicMock()
    llm_client.chat = AsyncMock(return_value="pong")
    conversation = ConversationStore(max_messages=20)
    # Предзаполняем историю двумя сообщениями, чтобы после handler'а длина
    # достигла порога (threshold=4 > SUMMARY_KEPT_TAIL=2 — суммаризация
    # реально запустится).
    conversation.add_user_message(user_id=5, content="hi")
    conversation.add_assistant_message(user_id=5, content="hello")
    summarizer = MagicMock()
    summarizer.summarize = AsyncMock(side_effect=LLMTimeout("boom"))
    settings = _make_settings(history_summary_threshold=4)
    message = _make_message(text="ping", user_id=5)

    caplog.set_level(logging.WARNING, logger="app.handlers.messages")

    await handle_text(
        message, llm_client, registry, conversation, summarizer, settings
    )

    # Основной ответ всё равно отправлен пользователю.
    message.answer.assert_awaited_once_with("pong")
    # Суммаризатор был вызван и упал → есть WARNING.
    summarizer.summarize.assert_awaited_once()
    assert any(
        "summarize failed" in rec.message and rec.levelname == "WARNING"
        for rec in caplog.records
    )
    # История осталась as-is (без замены на резюме).
    assert conversation.get_history(user_id=5) == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "ping"},
        {"role": "assistant", "content": "pong"},
    ]


# --- старые сценарии: ошибки LLM, разбивка, лимит ввода ---


async def test_timeout_answers_about_slowness_and_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    registry = _make_registry()
    llm_client = MagicMock()
    llm_client.chat = AsyncMock(side_effect=LLMTimeout("t"))
    conversation = ConversationStore(max_messages=20)
    summarizer = _make_summarizer(MagicMock())
    settings = _make_settings()
    message = _make_message(text="ping")

    caplog.set_level(logging.WARNING, logger="app.handlers.messages")

    await handle_text(
        message, llm_client, registry, conversation, summarizer, settings
    )

    text = message.answer.call_args.args[0]
    assert "долго" in text
    assert any(record.levelname == "WARNING" for record in caplog.records)


async def test_unavailable_answers_about_downtime_and_errors(
    caplog: pytest.LogCaptureFixture,
) -> None:
    registry = _make_registry()
    llm_client = MagicMock()
    llm_client.chat = AsyncMock(side_effect=LLMUnavailable("down"))
    conversation = ConversationStore(max_messages=20)
    summarizer = _make_summarizer(MagicMock())
    settings = _make_settings()
    message = _make_message(text="ping")

    caplog.set_level(logging.ERROR, logger="app.handlers.messages")

    await handle_text(
        message, llm_client, registry, conversation, summarizer, settings
    )

    text = message.answer.call_args.args[0]
    assert "недоступна" in text
    assert any(record.levelname == "ERROR" for record in caplog.records)


async def test_bad_response_uses_exception_message_for_model_not_found() -> None:
    registry = _make_registry()
    llm_client = MagicMock()
    llm_client.chat = AsyncMock(side_effect=LLMBadResponse("модель не найдена"))
    conversation = ConversationStore(max_messages=20)
    summarizer = _make_summarizer(MagicMock())
    settings = _make_settings()
    message = _make_message(text="ping")

    await handle_text(
        message, llm_client, registry, conversation, summarizer, settings
    )

    text = message.answer.call_args.args[0]
    assert "/models" in text or "не найдена" in text


async def test_generic_llm_error_returns_generic_message() -> None:
    registry = _make_registry()
    llm_client = MagicMock()
    llm_client.chat = AsyncMock(side_effect=LLMError("unknown"))
    conversation = ConversationStore(max_messages=20)
    summarizer = _make_summarizer(MagicMock())
    settings = _make_settings()
    message = _make_message(text="ping")

    await handle_text(
        message, llm_client, registry, conversation, summarizer, settings
    )

    text = message.answer.call_args.args[0]
    assert "ошибка" in text.lower()


async def test_long_response_is_split_into_chunks() -> None:
    registry = _make_registry()
    llm_client = MagicMock()
    long_response = "x" * 10000  # > 4096
    llm_client.chat = AsyncMock(return_value=long_response)
    conversation = ConversationStore(max_messages=20)
    summarizer = _make_summarizer(MagicMock())
    settings = _make_settings()
    message = _make_message(text="ping")

    await handle_text(
        message, llm_client, registry, conversation, summarizer, settings
    )

    assert message.answer.await_count >= 3


async def test_too_long_input_is_rejected_without_llm_call() -> None:
    registry = _make_registry()
    llm_client = MagicMock()
    llm_client.chat = AsyncMock()
    conversation = ConversationStore(max_messages=20)
    summarizer = _make_summarizer(MagicMock())
    settings = _make_settings()
    too_long = "x" * (MAX_INPUT_LENGTH + 1)
    message = _make_message(text=too_long, user_id=11)

    await handle_text(
        message, llm_client, registry, conversation, summarizer, settings
    )

    llm_client.chat.assert_not_awaited()
    text = message.answer.call_args.args[0]
    assert "сократите" in text.lower() or "слишком" in text.lower()
    # История не загрязнена слишком длинным сообщением.
    assert conversation.get_history(user_id=11) == []


async def test_handler_logs_user_chat_model_dur_ms(
    caplog: pytest.LogCaptureFixture,
) -> None:
    registry = _make_registry()
    registry.set_model(42, "deepseek-r1:1.5b")
    llm_client = MagicMock()
    llm_client.chat = AsyncMock(return_value="ok")
    conversation = ConversationStore(max_messages=20)
    summarizer = _make_summarizer(MagicMock())
    settings = _make_settings()
    message = _make_message(text="hi", user_id=42, chat_id=77)

    caplog.set_level(logging.INFO, logger="app.handlers.messages")

    await handle_text(
        message, llm_client, registry, conversation, summarizer, settings
    )

    assert any(
        "user=42" in rec.message
        and "chat=77" in rec.message
        and "model=deepseek-r1:1.5b" in rec.message
        and "dur_ms=" in rec.message
        for rec in caplog.records
        if rec.message.startswith("message ")
    )
