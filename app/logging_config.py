"""Единая настройка логирования приложения.

Конфигурируется через `logging.config.dictConfig`: консольный вывод и файл с ротацией
(см. `_docs/stack.md` §5, `_docs/instructions.md` §5). Секреты в логи не пишутся.
"""

from __future__ import annotations

import logging.config
from pathlib import Path

from app.config import Settings

_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def setup_logging(settings: Settings) -> None:
    """Применить конфигурацию логирования по настройкам.

    - Создаёт каталог для файла логов (при необходимости).
    - Настраивает два handler'а: `console` (StreamHandler) и `file` (RotatingFileHandler).
    - Формат: timestamp | level | logger | message.
    """
    log_path = Path(settings.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    level = settings.log_level.upper()

    config: dict = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {"format": _LOG_FORMAT},
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": level,
                "formatter": "standard",
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": level,
                "formatter": "standard",
                "filename": str(log_path),
                "maxBytes": 1_000_000,
                "backupCount": 3,
                "encoding": "utf-8",
            },
        },
        "root": {
            "level": level,
            "handlers": ["console", "file"],
        },
    }
    logging.config.dictConfig(config)
