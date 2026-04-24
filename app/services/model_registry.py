"""In-memory реестр пользовательских настроек (выбранная модель и системный промпт).

Хранит только runtime-настройки пользователя (`user_id -> model`, `user_id -> prompt`),
истории диалогов не хранит (требование ТЗ: stateless, без БД — см. `docs/architecture.md` §3.5,
`docs/requirements.md`).

Потокобезопасность: операции — обычные dict-ассайны/чтения. В CPython они атомарны
благодаря GIL, а данные пользователя изолированы по ключу `user_id`. Конкурентные
handler'ы aiogram работают в одном event loop'е, так что lock не требуется.
"""

from __future__ import annotations


class UserSettingsRegistry:
    """Хранит выбранную пользователем модель и системный промпт.

    Пользователи, не устанавливавшие значение, получают default из конструктора.
    """

    def __init__(self, *, default_model: str, default_prompt: str) -> None:
        self._default_model = default_model
        self._default_prompt = default_prompt
        self._models: dict[int, str] = {}
        self._prompts: dict[int, str] = {}

    # --- model ---

    def get_model(self, user_id: int) -> str:
        """Вернуть модель пользователя, иначе default."""
        return self._models.get(user_id, self._default_model)

    def set_model(self, user_id: int, model: str) -> None:
        """Установить модель пользователя."""
        self._models[user_id] = model

    def reset_model(self, user_id: int) -> None:
        """Сбросить модель пользователя к default."""
        self._models.pop(user_id, None)

    # --- prompt ---

    def get_prompt(self, user_id: int) -> str:
        """Вернуть системный промпт пользователя, иначе default."""
        return self._prompts.get(user_id, self._default_prompt)

    def set_prompt(self, user_id: int, prompt: str) -> None:
        """Установить системный промпт пользователя."""
        self._prompts[user_id] = prompt

    def reset_prompt(self, user_id: int) -> None:
        """Сбросить системный промпт пользователя к default."""
        self._prompts.pop(user_id, None)

    # --- combined ---

    def reset(self, user_id: int) -> None:
        """Сбросить оба значения пользователя к default."""
        self.reset_model(user_id)
        self.reset_prompt(user_id)
