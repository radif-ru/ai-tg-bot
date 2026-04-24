"""Тесты настройки логирования."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest


def test_setup_logging_creates_file_and_dir(
    monkeypatch: pytest.MonkeyPatch,
    base_env: dict[str, str],
    tmp_path: Path,
) -> None:
    log_file = tmp_path / "nested" / "bot.log"
    monkeypatch.setenv("LOG_FILE", str(log_file))

    from app.config import Settings
    from app.logging_config import setup_logging

    settings = Settings(_env_file=None)
    setup_logging(settings)

    logger = logging.getLogger("ai_tg_bot.test")
    logger.info("hello-from-test")
    for handler in logging.getLogger().handlers:
        handler.flush()

    assert log_file.exists()
    content = log_file.read_text(encoding="utf-8")
    assert "hello-from-test" in content


def test_setup_logging_does_not_leak_token(
    monkeypatch: pytest.MonkeyPatch,
    base_env: dict[str, str],
    tmp_path: Path,
) -> None:
    token = "SECRET-TOKEN-12345"
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", token)
    log_file = tmp_path / "bot.log"
    monkeypatch.setenv("LOG_FILE", str(log_file))
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")

    from app.config import Settings
    from app.logging_config import setup_logging

    settings = Settings(_env_file=None)
    setup_logging(settings)

    logger = logging.getLogger("ai_tg_bot.secret_probe")
    logger.debug("settings=%s", settings)
    logger.debug("repr=%r", settings)
    for handler in logging.getLogger().handlers:
        handler.flush()

    content = log_file.read_text(encoding="utf-8")
    assert token not in content
