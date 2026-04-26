"""Клиент локальной LLM (Ollama) и иерархия ошибок.

Реализация поверх `ollama.AsyncClient` (async) — выбор зафиксирован здесь,
альтернатива — прямые HTTP-запросы через `httpx.AsyncClient` (см. _docs/stack.md §3).

Модуль не должен импортировать `aiogram` — это сервис-слой, изолированный от Telegram.
"""

from __future__ import annotations

import asyncio
import logging
import time

import httpx
import ollama

__all__ = [
    "LLMError",
    "LLMTimeout",
    "LLMUnavailable",
    "LLMBadResponse",
    "OllamaClient",
    "estimate_tokens",
]

_logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Базовое исключение LLM-слоя."""


class LLMTimeout(LLMError):
    """Модель не успела ответить за отведённый таймаут."""


class LLMUnavailable(LLMError):
    """LLM-сервис (Ollama) недоступен (сетевая ошибка, нет соединения)."""


class LLMBadResponse(LLMError):
    """LLM вернула некорректный ответ (HTTP 4xx/5xx, пустой текст и т. п.)."""


class OllamaClient:
    """Асинхронный клиент Ollama.

    Все сетевые/HTTP-ошибки маппятся в иерархию `LLMError`.
    На каждый вызов пишется INFO-запись с метриками `model`, `len_in`, `len_out`,
    `dur_ms`, `status`.
    """

    def __init__(
        self,
        base_url: str,
        timeout: float,
        client: ollama.AsyncClient | None = None,
    ) -> None:
        self._client = client if client is not None else ollama.AsyncClient(
            host=base_url, timeout=timeout
        )

    async def generate(
        self,
        prompt: str,
        *,
        model: str,
        system_prompt: str | None,
    ) -> str:
        """Сгенерировать ответ по prompt.

        :raises LLMTimeout: при таймауте.
        :raises LLMUnavailable: при отсутствии соединения с Ollama.
        :raises LLMBadResponse: при HTTP-ошибке (404/5xx) или пустом ответе.
        :raises LLMError: при иной ошибке LLM-слоя.
        """
        started = time.monotonic()
        status = "ok"
        text = ""
        try:
            resp = await self._client.generate(
                model=model,
                prompt=prompt,
                system=system_prompt,
                stream=False,
            )
            text = getattr(resp, "response", "") or ""
            if not text.strip():
                status = "empty"
                raise LLMBadResponse("LLM вернула пустой ответ")
            return text
        except httpx.TimeoutException as exc:
            status = "timeout"
            raise LLMTimeout("таймаут запроса к LLM") from exc
        except asyncio.TimeoutError as exc:
            status = "timeout"
            raise LLMTimeout("таймаут запроса к LLM") from exc
        except httpx.ConnectError as exc:
            status = "unavailable"
            raise LLMUnavailable("LLM недоступна (нет соединения)") from exc
        except ollama.ResponseError as exc:
            code = int(getattr(exc, "status_code", -1))
            status = f"http_{code}"
            if code == 404:
                raise LLMBadResponse("модель не найдена") from exc
            raise LLMBadResponse(
                f"LLM вернула ошибку HTTP {code}: {getattr(exc, 'error', exc)}"
            ) from exc
        except LLMError:
            raise
        except Exception as exc:  # noqa: BLE001 — финальный защитный маппинг
            status = "error"
            raise LLMError(f"неожиданная ошибка LLM: {exc}") from exc
        finally:
            dur_ms = int((time.monotonic() - started) * 1000)
            _logger.info(
                "llm model=%s len_in=%d len_out=%d dur_ms=%d status=%s",
                model,
                len(prompt),
                len(text),
                dur_ms,
                status,
            )

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
    ) -> str:
        """Сгенерировать ответ по полной истории сообщений (chat-API).

        `messages` — список словарей `{"role": "system" | "user" | "assistant",
        "content": "..."}` в хронологическом порядке. Маппинг ошибок —
        идентичен `generate()`.

        :raises LLMTimeout: при таймауте.
        :raises LLMUnavailable: при отсутствии соединения с Ollama.
        :raises LLMBadResponse: при HTTP-ошибке (404/5xx) или пустом ответе.
        :raises LLMError: при иной ошибке LLM-слоя.
        """
        started = time.monotonic()
        status = "ok"
        text = ""
        len_in = sum(len(m.get("content", "")) for m in messages)
        try:
            resp = await self._client.chat(
                model=model,
                messages=messages,
                stream=False,
            )
            message = getattr(resp, "message", None)
            if message is None and isinstance(resp, dict):
                message = resp.get("message")
            if message is None:
                content = ""
            else:
                content = getattr(message, "content", None)
                if content is None and isinstance(message, dict):
                    content = message.get("content")
            text = content or ""
            if not text.strip():
                status = "empty"
                raise LLMBadResponse("LLM вернула пустой ответ")
            return text
        except httpx.TimeoutException as exc:
            status = "timeout"
            raise LLMTimeout("таймаут запроса к LLM") from exc
        except asyncio.TimeoutError as exc:
            status = "timeout"
            raise LLMTimeout("таймаут запроса к LLM") from exc
        except httpx.ConnectError as exc:
            status = "unavailable"
            raise LLMUnavailable("LLM недоступна (нет соединения)") from exc
        except ollama.ResponseError as exc:
            code = int(getattr(exc, "status_code", -1))
            status = f"http_{code}"
            if code == 404:
                raise LLMBadResponse("модель не найдена") from exc
            raise LLMBadResponse(
                f"LLM вернула ошибку HTTP {code}: {getattr(exc, 'error', exc)}"
            ) from exc
        except LLMError:
            raise
        except Exception as exc:  # noqa: BLE001 — финальный защитный маппинг
            status = "error"
            raise LLMError(f"неожиданная ошибка LLM: {exc}") from exc
        finally:
            dur_ms = int((time.monotonic() - started) * 1000)
            _logger.info(
                "llm.chat model=%s messages=%d len_in=%d len_out=%d dur_ms=%d status=%s",
                model,
                len(messages),
                len_in,
                len(text),
                dur_ms,
                status,
            )

    async def close(self) -> None:
        """Закрыть нижележащий HTTP-клиент (graceful shutdown)."""
        inner = getattr(self._client, "_client", None)
        if inner is not None:
            aclose = getattr(inner, "aclose", None)
            if aclose is not None:
                await aclose()


def estimate_tokens(value: str | list[dict[str, str]]) -> int:
    """Грубая оценка количества токенов для логирования размера контекста.

    Эвристика: `ceil(total_chars / 4)`. Это **не точный токенайзер** — для
    логов и порогов хватает, но реальное число токенов локальных моделей
    (особенно для кириллицы) может отличаться в 1.5–2 раза. Точный подсчёт
    через `tiktoken` / HF — кандидат `_docs/roadmap.md`.

    :param value: строка или список сообщений (`{"role", "content"}`); во
        втором случае суммируются длины `content`.
    :return: оценка количества токенов, всегда >= 1.
    """
    if isinstance(value, str):
        chars = len(value)
    else:
        chars = sum(len(m.get("content", "")) for m in value)
    return max(1, (chars + 3) // 4)
