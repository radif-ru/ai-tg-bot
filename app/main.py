"""Точка входа бота: сборка Bot/Dispatcher, регистрация роутеров, запуск polling."""

from __future__ import annotations

import logging

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.types import BotCommand

from app.config import Settings
from app.handlers import commands as commands_handlers
from app.logging_config import setup_logging
from app.services.llm import OllamaClient
from app.services.model_registry import UserSettingsRegistry


async def main() -> None:
    """Собрать приложение и запустить long-polling.

    Последовательно:
    - Загружает `Settings` из окружения / `.env`.
    - Настраивает логирование.
    - Создаёт `Bot`, `Dispatcher`, LLM-клиент и registry.
    - Прокидывает зависимости в `workflow_data` диспетчера (DI aiogram 3).
    - Регистрирует роутеры и команды BotFather.
    - Запускает polling, в finally закрывает клиенты.
    """
    settings = Settings()
    setup_logging(settings)
    logger = logging.getLogger(__name__)

    bot = Bot(
        token=settings.telegram_bot_token.get_secret_value(),
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )
    dispatcher = Dispatcher()

    llm_client = OllamaClient(
        base_url=settings.ollama_base_url,
        timeout=settings.ollama_timeout,
    )
    registry = UserSettingsRegistry(
        default_model=settings.ollama_default_model,
        default_prompt=settings.system_prompt,
    )

    dispatcher["settings"] = settings
    dispatcher["llm_client"] = llm_client
    dispatcher["registry"] = registry

    dispatcher.include_router(commands_handlers.router)

    await bot.set_my_commands(
        [
            BotCommand(command="start", description="Начать работу"),
            BotCommand(command="help", description="Справка"),
            BotCommand(command="models", description="Список моделей"),
            BotCommand(command="model", description="Выбрать модель"),
            BotCommand(command="prompt", description="Задать системный промпт"),
        ]
    )

    logger.info("Bot started")

    try:
        await dispatcher.start_polling(bot)
    finally:
        await llm_client.close()
        await bot.session.close()
