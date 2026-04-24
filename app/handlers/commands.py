"""Handler'ы команд бота (/start, /help и др.).

См. `docs/commands.md` — спецификация поведения каждой команды.
"""

from __future__ import annotations

from html import escape

from aiogram import Router
from aiogram.filters import Command, CommandStart
from aiogram.types import Message

from app.services.model_registry import UserSettingsRegistry

router = Router(name="commands")

_PROMPT_PREVIEW_LIMIT = 200

START_TEXT = (
    "Привет! Я — AI-бот на локальной LLM (Ollama).\n"
    "\n"
    "Просто напиши мне что угодно — я отвечу.\n"
    "\n"
    "Команды:\n"
    "/help — справка\n"
    "/models — доступные модели\n"
    "/model &lt;имя&gt; — выбрать модель\n"
    "/prompt &lt;текст&gt; — задать системный промпт (без текста — сброс)"
)


@router.message(CommandStart())
async def cmd_start(message: Message) -> None:
    """Ответить приветствием и списком команд."""
    await message.answer(START_TEXT)


@router.message(Command("help"))
async def cmd_help(message: Message, registry: UserSettingsRegistry) -> None:
    """Вывести расширенную справку: текущая модель и текущий системный промпт."""
    user_id = message.from_user.id if message.from_user else 0
    current_model = registry.get_model(user_id)
    current_prompt = registry.get_prompt(user_id)

    if len(current_prompt) > _PROMPT_PREVIEW_LIMIT:
        prompt_preview = current_prompt[:_PROMPT_PREVIEW_LIMIT] + "…"
    else:
        prompt_preview = current_prompt

    text = (
        "<b>Справка</b>\n"
        "\n"
        f"• Активная модель: <code>{escape(current_model)}</code>\n"
        f"• Текущий системный промпт: <i>{escape(prompt_preview)}</i>\n"
        "\n"
        "<b>Команды</b>\n"
        "/start — приветствие\n"
        "/help — это сообщение\n"
        "/models — список доступных моделей\n"
        "/model &lt;имя&gt; — сменить модель\n"
        "/prompt &lt;текст&gt; — задать системный промпт (без аргумента — сброс)"
    )
    await message.answer(text)
