"""Handler'ы команд бота (/start, /help, /models, /model, /prompt).

См. `docs/commands.md` — спецификация поведения каждой команды.
"""

from __future__ import annotations

from html import escape

from aiogram import Router
from aiogram.filters import Command, CommandObject, CommandStart
from aiogram.types import Message

from app.config import Settings
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


@router.message(Command("models"))
async def cmd_models(
    message: Message,
    settings: Settings,
    registry: UserSettingsRegistry,
) -> None:
    """Показать список доступных моделей с отметкой активной."""
    user_id = message.from_user.id if message.from_user else 0
    active = registry.get_model(user_id)

    lines = ["<b>Доступные модели</b>"]
    for name in settings.ollama_available_models:
        marker = " ← активная" if name == active else ""
        lines.append(f"• <code>{escape(name)}</code>{marker}")
    lines.append("")
    lines.append("Сменить: <code>/model &lt;имя&gt;</code>")

    await message.answer("\n".join(lines))


@router.message(Command("model"))
async def cmd_model(
    message: Message,
    command: CommandObject,
    settings: Settings,
    registry: UserSettingsRegistry,
) -> None:
    """Переключить активную модель пользователя."""
    user_id = message.from_user.id if message.from_user else 0
    raw_arg = (command.args or "").strip()

    if not raw_arg:
        await message.answer(
            "Использование: <code>/model &lt;имя&gt;</code>. Список: /models"
        )
        return

    name = raw_arg.split()[0]
    if name not in settings.ollama_available_models:
        available = ", ".join(
            f"<code>{escape(m)}</code>" for m in settings.ollama_available_models
        )
        await message.answer(f"Модель не найдена. Доступно: {available}")
        return

    registry.set_model(user_id, name)
    await message.answer(f"Модель переключена на <code>{escape(name)}</code>.")


@router.message(Command("prompt"))
async def cmd_prompt(
    message: Message,
    command: CommandObject,
    registry: UserSettingsRegistry,
) -> None:
    """Установить системный промпт. Без аргумента — сброс к default."""
    user_id = message.from_user.id if message.from_user else 0
    arg = (command.args or "").strip()

    if not arg:
        registry.reset_prompt(user_id)
        await message.answer("Системный промпт сброшен к значению по умолчанию.")
        return

    registry.set_prompt(user_id, arg)
    await message.answer("Системный промпт обновлён.")
