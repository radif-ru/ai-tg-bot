"""In-memory история диалога per-user.

Хранит для каждого `user_id` список сообщений в формате
`[{"role": "user" | "assistant" | "system", "content": "..."}, ...]`.
Лимит на размер истории — `max_messages`; при превышении самые старые сообщения
удаляются (FIFO). Резюме (`role: system`), которое появляется в результате
суммаризации, считается обычным сообщением для целей лимита.

Потокобезопасность: операции — обычные dict/list-ассайны/чтения. В CPython они
атомарны благодаря GIL, а данные пользователя изолированы по ключу `user_id`.
Конкурентные handler'ы aiogram работают в одном event loop'е, так что lock
не требуется. Тот же подход — в `app/services/model_registry.py`.

Класс не зависит ни от `aiogram`, ни от `ollama`: это сервис-слой.
"""

from __future__ import annotations

__all__ = ["ConversationStore"]

# Префикс, которым помечается сжатая (саммаризированная) часть истории.
# Выносится в константу, чтобы handler и тесты ссылались на одно и то же.
SUMMARY_PREFIX = "Краткое резюме предыдущей части диалога: "


class ConversationStore:
    """Хранит историю диалога per-user в памяти процесса.

    История ограничена по количеству сообщений: при превышении `max_messages`
    самые старые удаляются (FIFO). Хранятся только пары `role`/`content` —
    ничего лишнего из update'а Telegram сюда не попадает.

    После рестарта процесса вся история теряется (CON-1: персистентного
    хранилища нет, in-memory — допустимо).
    """

    def __init__(self, *, max_messages: int) -> None:
        if max_messages <= 0:
            raise ValueError(
                f"max_messages must be > 0, got {max_messages}"
            )
        self._max_messages = max_messages
        self._histories: dict[int, list[dict[str, str]]] = {}

    def get_history(self, user_id: int) -> list[dict[str, str]]:
        """Вернуть копию истории пользователя; пустой список, если истории нет.

        Возвращается **копия**, чтобы handler не мог случайно мутировать
        внутреннее состояние (мы не доверяем callers, см. instructions.md
        § «Стиль кода»).
        """
        history = self._histories.get(user_id)
        if history is None:
            return []
        return [dict(message) for message in history]

    def add_user_message(self, user_id: int, content: str) -> None:
        """Добавить пользовательское сообщение в историю + усечь по лимиту."""
        self._append(user_id, role="user", content=content)

    def add_assistant_message(self, user_id: int, content: str) -> None:
        """Добавить ответ ассистента в историю + усечь по лимиту."""
        self._append(user_id, role="assistant", content=content)

    def replace_with_summary(
        self, user_id: int, summary: str, *, kept_tail: int
    ) -> None:
        """Заменить старую часть истории одним system-сообщением с резюме.

        Сохраняет последние `kept_tail` сообщений как есть, всё, что было
        раньше них, заменяется одним `{"role": "system", "content": SUMMARY_PREFIX + summary}`.

        Если `kept_tail >= len(history)` — заменять нечего, no-op.
        Если у пользователя нет истории — no-op.
        """
        if kept_tail < 0:
            raise ValueError(f"kept_tail must be >= 0, got {kept_tail}")
        history = self._histories.get(user_id)
        if not history:
            return
        if kept_tail >= len(history):
            return
        tail = history[-kept_tail:] if kept_tail > 0 else []
        summary_message = {
            "role": "system",
            "content": f"{SUMMARY_PREFIX}{summary}",
        }
        self._histories[user_id] = [summary_message, *tail]
        self._truncate(user_id)

    def clear(self, user_id: int) -> None:
        """Полностью удалить историю пользователя."""
        self._histories.pop(user_id, None)

    # --- internal ---

    def _append(self, user_id: int, *, role: str, content: str) -> None:
        history = self._histories.setdefault(user_id, [])
        history.append({"role": role, "content": content})
        self._truncate(user_id)

    def _truncate(self, user_id: int) -> None:
        history = self._histories.get(user_id)
        if history is None:
            return
        overflow = len(history) - self._max_messages
        if overflow > 0:
            del history[:overflow]
