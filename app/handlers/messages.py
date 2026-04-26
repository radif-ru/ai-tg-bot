"""Handler произвольного текста: пользовательское сообщение → LLM → ответ.

Алгоритм (см. `_docs/architecture.md` §4 и `_docs/commands.md` § «Произвольный текст»):

1. Длина ввода проверяется (`MAX_INPUT_LENGTH`).
2. Сообщение пользователя дописывается в `ConversationStore`.
3. Контекст = `[system] + history` логируется (`LOG_LLM_CONTEXT` управляет
   тем, печатать ли сам payload, или только размеры).
4. Контекст отправляется в `OllamaClient.chat(messages, model=...)`.
5. Ответ ассистента дописывается в историю.
6. Если длина истории достигла `HISTORY_SUMMARY_THRESHOLD` —
   старая часть истории сжимается через `Summarizer` и заменяется
   одним system-сообщением с резюме (см. `ConversationStore.replace_with_summary`).
   Ошибка суммаризации не валит ответ пользователю — пишется WARNING,
   история оставляется как есть и обрежется естественным FIFO.
7. Ответ пользователю отправляется (с разбивкой по лимиту Telegram).
"""

from __future__ import annotations

import json
import logging
import time

from aiogram import F, Router
from aiogram.enums import ChatAction
from aiogram.types import Message

from app.config import Settings
from app.services.conversation import ConversationStore
from app.services.llm import (
    LLMBadResponse,
    LLMError,
    LLMTimeout,
    LLMUnavailable,
    OllamaClient,
    estimate_tokens,
)
from app.services.model_registry import UserSettingsRegistry
from app.services.summarizer import Summarizer
from app.utils.text import TELEGRAM_MESSAGE_LIMIT, split_long_message

router = Router(name="messages")

MAX_INPUT_LENGTH = 4000
# Сколько последних сообщений сохранить «как есть» при суммаризации.
# 2 = одна последняя пара user+assistant (или просто последний user, если
# суммаризация запускается до записи assistant-ответа — в нашем pipeline
# это уже после, см. функцию ниже).
SUMMARY_KEPT_TAIL = 2

_logger = logging.getLogger(__name__)


@router.message(F.text & ~F.text.startswith("/"))
async def handle_text(
    message: Message,
    llm_client: OllamaClient,
    registry: UserSettingsRegistry,
    conversation: ConversationStore,
    summarizer: Summarizer,
    settings: Settings,
) -> None:
    """Обработать произвольный текст пользователя: LLM-запрос с историей → ответ."""
    user_id = message.from_user.id if message.from_user else 0
    chat_id = message.chat.id
    text = message.text or ""

    if len(text) > MAX_INPUT_LENGTH:
        await message.answer(
            f"Слишком длинный запрос, сократите (лимит — {MAX_INPUT_LENGTH} символов)."
        )
        return

    model = registry.get_model(user_id)
    system_prompt = registry.get_prompt(user_id)

    conversation.add_user_message(user_id, text)
    history = conversation.get_history(user_id)
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        *history,
    ]

    _log_context(
        user_id=user_id,
        chat_id=chat_id,
        model=model,
        messages=messages,
        log_payload=settings.log_llm_context,
    )

    started = time.monotonic()
    status = "ok"
    response: str | None = None

    try:
        await message.bot.send_chat_action(chat_id, ChatAction.TYPING)
        response = await llm_client.chat(messages, model=model)
    except LLMTimeout:
        status = "timeout"
        _logger.warning(
            "LLM timeout user=%s chat=%s model=%s", user_id, chat_id, model
        )
        await message.answer("Модель слишком долго отвечает. Попробуйте ещё раз.")
    except LLMUnavailable:
        status = "unavailable"
        _logger.error(
            "LLM unavailable user=%s chat=%s model=%s", user_id, chat_id, model
        )
        await message.answer("LLM сейчас недоступна, попробуйте позже.")
    except LLMBadResponse as exc:
        status = "bad_response"
        _logger.error(
            "LLM bad response user=%s chat=%s model=%s err=%s",
            user_id,
            chat_id,
            model,
            exc,
        )
        msg = str(exc)
        if "не найдена" in msg or "not found" in msg.lower():
            await message.answer("Модель не найдена, выберите через /models.")
        else:
            await message.answer(msg or "Произошла ошибка при обращении к LLM.")
    except LLMError:
        status = "error"
        _logger.exception(
            "LLM error user=%s chat=%s model=%s", user_id, chat_id, model
        )
        await message.answer("Произошла ошибка при обращении к LLM.")
    finally:
        dur_ms = int((time.monotonic() - started) * 1000)
        _logger.info(
            "message user=%s chat=%s model=%s len_in=%d dur_ms=%d status=%s",
            user_id,
            chat_id,
            model,
            len(text),
            dur_ms,
            status,
        )

    if response is None:
        return

    conversation.add_assistant_message(user_id, response)
    await _maybe_summarize(
        conversation=conversation,
        summarizer=summarizer,
        user_id=user_id,
        model=model,
        threshold=settings.history_summary_threshold,
    )

    for chunk in split_long_message(response, limit=TELEGRAM_MESSAGE_LIMIT):
        await message.answer(chunk)


def _log_context(
    *,
    user_id: int,
    chat_id: int,
    model: str,
    messages: list[dict[str, str]],
    log_payload: bool,
) -> None:
    """Записать в лог контекст перед LLM-запросом + его размер.

    `log_payload=True` — печатается полный JSON сообщений (требование ТЗ §5);
    `log_payload=False` — только размеры (`messages=`, `tokens=`).
    """
    tokens = estimate_tokens(messages)
    if log_payload:
        _logger.info(
            "llm_context user=%s chat=%s model=%s messages=%d tokens=%d payload=%s",
            user_id,
            chat_id,
            model,
            len(messages),
            tokens,
            json.dumps(messages, ensure_ascii=False),
        )
    else:
        _logger.info(
            "llm_context user=%s chat=%s model=%s messages=%d tokens=%d",
            user_id,
            chat_id,
            model,
            len(messages),
            tokens,
        )


async def _maybe_summarize(
    *,
    conversation: ConversationStore,
    summarizer: Summarizer,
    user_id: int,
    model: str,
    threshold: int,
) -> None:
    """Если история достигла порога — сжать её и заменить старую часть резюме.

    Любая `LLMError` от суммаризации **не валит** ответ пользователю:
    логируется WARNING, история остаётся как есть и обрежется FIFO в
    `ConversationStore._truncate` при следующих add'ах.
    """
    history = conversation.get_history(user_id)
    if len(history) < threshold:
        return
    if len(history) <= SUMMARY_KEPT_TAIL:
        # Защита от вырожденного случая: threshold <= SUMMARY_KEPT_TAIL.
        return

    to_summarize = history[:-SUMMARY_KEPT_TAIL]

    started = time.monotonic()
    try:
        summary = await summarizer.summarize(to_summarize, model=model)
    except LLMError as exc:
        _logger.warning(
            "summarize failed user=%s model=%s err=%s",
            user_id,
            model,
            exc,
        )
        return

    conversation.replace_with_summary(
        user_id, summary, kept_tail=SUMMARY_KEPT_TAIL
    )
    dur_ms = int((time.monotonic() - started) * 1000)
    _logger.info(
        "summarized history user=%s len_before=%d len_after=%d dur_ms=%d",
        user_id,
        len(history),
        len(conversation.get_history(user_id)),
        dur_ms,
    )
