"""Настройки приложения, читаемые из переменных окружения / `.env`.

Все секреты и конфигурация должны приходить только сюда (см. `_docs/instructions.md` §6).
"""

from __future__ import annotations

from typing import Annotated

from pydantic import SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    """Конфигурация приложения.

    Загружается из переменных окружения (с поддержкой файла `.env`).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    telegram_bot_token: SecretStr
    ollama_base_url: str = "http://localhost:11434"
    ollama_default_model: str
    ollama_available_models: Annotated[list[str], NoDecode]
    ollama_timeout: int = 60
    system_prompt: str = "Ты — полезный ассистент. Отвечай кратко и по делу."
    log_level: str = "INFO"
    log_file: str = "logs/bot.log"

    history_max_messages: int = 20
    history_summary_threshold: int = 10
    summarization_prompt: str = (
        "Кратко и точно резюмируй ключевые факты и решения из этого диалога "
        "в 2–4 предложениях. Ответ — только текст резюме, без вступлений."
    )
    log_llm_context: bool = True

    @field_validator("ollama_available_models", mode="before")
    @classmethod
    def _split_csv(cls, value: object) -> object:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @model_validator(mode="after")
    def _default_model_must_be_available(self) -> "Settings":
        if self.ollama_default_model not in self.ollama_available_models:
            raise ValueError(
                f"OLLAMA_DEFAULT_MODEL={self.ollama_default_model!r} "
                f"must be one of OLLAMA_AVAILABLE_MODELS={self.ollama_available_models}"
            )
        return self

    @model_validator(mode="after")
    def _history_limits_consistent(self) -> "Settings":
        if self.history_max_messages <= 0:
            raise ValueError(
                f"HISTORY_MAX_MESSAGES must be > 0, got {self.history_max_messages}"
            )
        if self.history_summary_threshold <= 0:
            raise ValueError(
                f"HISTORY_SUMMARY_THRESHOLD must be > 0, "
                f"got {self.history_summary_threshold}"
            )
        if self.history_summary_threshold > self.history_max_messages:
            raise ValueError(
                f"HISTORY_SUMMARY_THRESHOLD={self.history_summary_threshold} "
                f"must be <= HISTORY_MAX_MESSAGES={self.history_max_messages}"
            )
        return self
