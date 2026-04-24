"""Handler произвольного текста: пользовательское сообщение → LLM → ответ.

См. `docs/commands.md` §«Произвольный текст» и `docs/architecture.md` §4–§5.
"""

from __future__ import annotations

import logging
import time

from aiogram import F, Router
from aiogram.enums import ChatAction
from aiogram.types import Message

from app.services.llm import (
    LLMBadResponse,
    LLMError,
    LLMTimeout,
    LLMUnavailable,
    OllamaClient,
)
from app.services.model_registry import UserSettingsRegistry
from app.utils.text import TELEGRAM_MESSAGE_LIMIT, split_long_message

router = Router(name="messages")

MAX_INPUT_LENGTH = 4000

_logger = logging.getLogger(__name__)


@router.message(F.text & ~F.text.startswith("/"))
async def handle_text(
    message: Message,
    llm_client: OllamaClient,
    registry: UserSettingsRegistry,
) -> None:
    """Обработать произвольный текст пользователя: LLM-запрос → ответ."""
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

    started = time.monotonic()
    status = "ok"
    response: str | None = None

    try:
        await message.bot.send_chat_action(chat_id, ChatAction.TYPING)
        response = await llm_client.generate(
            text, model=model, system_prompt=system_prompt
        )
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

    if response is not None:
        for chunk in split_long_message(response, limit=TELEGRAM_MESSAGE_LIMIT):
            await message.answer(chunk)
