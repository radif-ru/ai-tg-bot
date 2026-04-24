"""Smoke-тест сборки приложения: main() доходит до 'Bot started' без сети."""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock

import pytest


async def test_main_logs_bot_started_and_closes(
    base_env: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    tmp_path,
) -> None:
    # Заменяем TELEGRAM_BOT_TOKEN и LOG_FILE, чтобы тест не писал в реальные логи.
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:fake")
    monkeypatch.setenv("LOG_FILE", str(tmp_path / "bot.log"))
    monkeypatch.setenv("OLLAMA_DEFAULT_MODEL", "m1")
    monkeypatch.setenv("OLLAMA_AVAILABLE_MODELS", "m1,m2")

    # Избегаем реальных сетевых вызовов:
    #   - set_my_commands — HTTP к Telegram;
    #   - start_polling — блокирующий long poll;
    #   - bot.session.close — async close TCP;
    #   - llm_client.close — async aclose httpx.
    from aiogram import Bot, Dispatcher
    from app.services.llm import OllamaClient

    monkeypatch.setattr(Bot, "set_my_commands", AsyncMock())
    monkeypatch.setattr(Dispatcher, "start_polling", AsyncMock())
    monkeypatch.setattr(OllamaClient, "close", AsyncMock())

    # setup_logging через dictConfig перетирает handler'ы root-логгера,
    # что ломает pytest caplog. В smoke-тесте логирование нам не нужно.
    import app.main as app_main

    monkeypatch.setattr(app_main, "setup_logging", lambda _settings: None)

    async def _fake_session_close(self):
        return None

    # bot.session.close — это AIOHTTP сессия, closure может срабатывать мгновенно.
    from aiogram.client.session.aiohttp import AiohttpSession

    monkeypatch.setattr(AiohttpSession, "close", _fake_session_close)

    caplog.set_level(logging.INFO, logger="app.main")

    from app.main import main

    await main()

    assert any("Bot started" in record.message for record in caplog.records)
