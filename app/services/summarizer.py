"""Сжатие старой части диалога через LLM (chat-API).

При достижении порога `Settings.history_summary_threshold` handler текста
вызывает `Summarizer.summarize(history)` и заменяет старую часть истории
одним system-сообщением с резюме (см. `app/services/conversation.py`).

Класс не зависит от `aiogram` — это сервис-слой. От `Settings` тоже не
зависит напрямую: промпт передаётся в конструктор (упрощает unit-тесты и
позволяет переопределить промпт без перезапуска приложения, если нужно).
"""

from __future__ import annotations

from app.services.llm import OllamaClient

__all__ = ["Summarizer"]


class Summarizer:
    """Обёртка над `OllamaClient.chat` для суммаризации истории диалога.

    Никаких retry / fallback: если LLM упала, исключение пробрасывается
    вызывающему коду (handler сам решает — оставить историю как есть и
    залогировать ошибку, или прервать ответ пользователю).
    """

    def __init__(self, client: OllamaClient, *, prompt: str) -> None:
        if not prompt or not prompt.strip():
            raise ValueError("summarization prompt must be non-empty")
        self._client = client
        self._prompt = prompt

    async def summarize(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
    ) -> str:
        """Запросить у LLM краткое резюме переданной истории.

        Формирует payload `[{"role": "system", "content": self._prompt}] + messages`
        и вызывает `client.chat(payload, model=model)`. Возвращает текст
        ответа без модификаций — handler сам обернёт его в формат
        «Краткое резюме …» через `ConversationStore.replace_with_summary`.

        :raises LLMError: пробрасывает любые ошибки `OllamaClient.chat`.
        """
        payload: list[dict[str, str]] = [
            {"role": "system", "content": self._prompt},
            *messages,
        ]
        return await self._client.chat(payload, model=model)
