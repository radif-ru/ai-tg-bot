# Текущее состояние проекта

Документ фиксирует фактическое состояние кода `app/` на момент написания: что работает, какие есть известные нюансы, ограничения и легаси. **При расхождении с кодом приоритет у кода**, документ должен быть подправлен следующим коммитом.

Читать обязательно перед любыми правками.

## 1. Что работает

- **Long polling** через `aiogram 3` (`app/main.py::main` → `dispatcher.start_polling(bot)`).
- **Команды** `/start`, `/help`, `/models`, `/model <name>`, `/prompt [<text>]`, `/reset` (`app/handlers/commands.py`).
- **Произвольный текст** → история → запрос в Ollama с полным контекстом → история → (опц.) суммаризация → ответ (`app/handlers/messages.py`).
- **LLM-клиент** на `ollama.AsyncClient` с двумя методами (`generate` и `chat(messages, model)`), идентичным маппингом ошибок в иерархию `LLMError` / `LLMTimeout` / `LLMUnavailable` / `LLMBadResponse` и функцией `estimate_tokens` (`app/services/llm.py`).
- **История диалога per-user** (in-memory) — `app/services/conversation.py::ConversationStore` (FIFO-обрезка по `Settings.history_max_messages`, `replace_with_summary`, `clear`).
- **Суммаризация истории** — `app/services/summarizer.py::Summarizer` (обёртка над `chat` с промптом из `Settings.summarization_prompt`); запускается в handler'е при `len(history) >= Settings.history_summary_threshold`, падение не валит ответ.
- **Логирование контекста** перед каждым LLM-запросом: `INFO llm_context user=… chat=… model=… messages=N tokens=K [payload=<JSON>]` (печать payload управляется `Settings.log_llm_context`).
- **Per-user runtime-настройки** (модель + системный промпт) — `app/services/model_registry.py::UserSettingsRegistry` (in-memory, без персистентности).
- **Конфиг** через `pydantic-settings` (`app/config.py::Settings`), валидация `OLLAMA_DEFAULT_MODEL ∈ OLLAMA_AVAILABLE_MODELS`.
- **Логирование** в консоль + файл с ротацией (`app/logging_config.py`), формат `%(asctime)s | %(levelname)s | %(name)s | %(message)s`.
- **Middleware логирования** апдейтов (`app/middlewares/logging_mw.py`) — пишет user/chat/type/dur_ms/status, без контента сообщения.
- **Глобальный error handler** (`app/handlers/errors.py`) — ловит необработанные исключения, отвечает пользователю нейтральным сообщением, не валит polling.
- **Разбивка длинных ответов** (> 4096 символов) — `app/utils/text.py::split_long_message`.
- **Тесты** — `pytest -q` (см. `tests/`), используют моки `aiogram` и `ollama`, без сетевых вызовов.

## 2. Известные проблемы и легаси

### 2.1 Нет throttling / rate-limit middleware

**Файл**: `app/middlewares/`. **Серьёзность**: низкая (для MVP не критично).

Любой пользователь может слать сообщения с любой частотой; это упрётся в скорость локальной модели Ollama, но создаёт риск bdoS своей же машины. Throttling на стороне бота отсутствует.

**Рекомендация**: простой `ThrottlingMiddleware` (1 сообщение / N секунд на пользователя). См. `_docs/roadmap.md` Этап 10, п.2.

### 2.2 Нет стриминга ответа Ollama

**Файл**: `app/services/llm.py::OllamaClient.generate / .chat`. **Серьёзность**: низкая.

Оба метода вызываются с `stream=False` — пользователь получает ответ целиком после полной генерации модели. Для медленных моделей это даёт ощущение «зависания» (помогает только индикатор «печатает…»).

**Рекомендация**: переключиться на `stream=True` и редактировать исходящее сообщение чанками. См. `_docs/roadmap.md` Этап 10, п.5.

### 2.3 Нет CI

**Файл**: репозиторий целиком. **Серьёзность**: низкая.

Нет `.github/workflows/*` с прогоном `pytest` и линтера. Тесты запускаются только локально.

**Рекомендация**: GitHub Actions: `setup-python` + `pip install -r requirements.txt` + `pytest -q`. См. `_docs/roadmap.md` Этап 10, п.4.

### 2.4 Нет Docker / docker-compose

**Файл**: репозиторий целиком. **Серьёзность**: низкая.

Запуск возможен только локально с предустановленной Ollama. Для деплоя на чужой машине нет образа `Dockerfile` и `docker-compose.yml` (бот + Ollama).

**Рекомендация**: см. `_docs/roadmap.md` Этап 10, п.3.

### 2.5 Линтер/форматтер не настроен

**Файл**: `pyproject.toml`. **Серьёзность**: низкая.

`ruff` упомянут в `_docs/stack.md` §7 как «рекомендовано», но не установлен и не сконфигурирован в `pyproject.toml`. Стиль кода поддерживается вручную.

**Рекомендация**: добавить `[tool.ruff]` блок и установить `ruff` в `requirements-dev.txt`.

### 2.6 Нет валидации `LOG_LEVEL`

**Файл**: `app/config.py::Settings.log_level`. **Серьёзность**: низкая.

`log_level: str = "INFO"` — любая строка проходит валидацию pydantic'ом. Если в `.env` написать `LOG_LEVEL=NOPE`, упадёт уже на `dictConfig` с не самой очевидной ошибкой.

**Рекомендация**: добавить `field_validator` или `Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]`.

### 2.7 Двойное логирование длительности в `messages.py`

**Файл**: `app/handlers/messages.py`. **Серьёзность**: низкая (избыточность, не баг).

`OllamaClient.chat` уже логирует `dur_ms` LLM-вызова, и `handle_text` логирует свой `dur_ms` поверх. На один пользовательский запрос пишется 2 INFO-строки про длительность (плюс одна от middleware) и ещё одна `llm_context` перед вызовом. Дубликатов в смысле метрик нет, но 3–4 записи на запрос — много для prod-ситуации.

**Рекомендация**: оставить как есть для MVP; при настройке observability — пересмотреть и оставить одну сводную строку.

## 3. Архитектурные нюансы (не баги, но знать обязательно)

- **In-memory per-user состояние, без персистентности**: выбранная модель, системный промпт **и** история диалога (включая суммаризированный контекст) живут только в памяти процесса. После рестарта всё возвращается к default из `Settings` (`_docs/requirements.md` §FR-3, §CON-1, §ASM-4).
- **Контекст LLM собирается на каждый запрос**: `[system] + conversation.get_history(user_id)`. `get_history` возвращает **копию** списка — handler не должен мутировать возвращённое.
- **Оценка размера контекста** — грубая: `estimate_tokens = max(1, len(text) // 4)`. Используется только для логирования (`messages=N tokens=K`), не для ограничения запроса.
- **Логирование полного payload** (`llm_context … payload=<JSON>`) может раздуть лог и попасть на чувствительные данные пользователя — управляется флагом `Settings.log_llm_context` (default `True`; в prod рекомендуется выставить `False`).
- **Суммаризация — best effort**: падение `Summarizer.summarize` (таймаут, недоступен Ollama, плохой ответ) логируется и **не блокирует** ответ пользователю; история остаётся как есть до следующего срабатывания порога.
- **Один общий `OllamaClient`** на всё приложение (создаётся в `main.py`, закрывается в `finally`). Не плодим клиенты в handler'ах.
- **Очерёдность роутеров** в `main.py`: `commands_handlers.router` → `messages_handlers.router` → `errors_handlers.router`. Команды должны идти раньше, чтобы текст вида `/start ...` не попал в обработчик произвольного текста.
- **Обработка длинных ответов**: handler `handle_text` сам режет ответ через `split_long_message`. Telegram обрежет всё, что > 4096, отдельной ошибкой `BadRequest` — это исключено резкой на стороне бота.
- **`parse_mode=ParseMode.HTML`** установлен по умолчанию (`DefaultBotProperties` в `main.py`). Все хендлеры должны экранировать пользовательский ввод (`html.escape`) перед вставкой — см. `commands.py`.

## 4. Что точно не сломано

Чтобы не паниковать без причины:

- `OllamaClient.close()` корректно закрывает подлежащий `httpx.AsyncClient` через `client._client.aclose()` (graceful shutdown).
- Глобальный `errors_handlers.router` ловит **всё**, не пойманное в handlers — polling не падает.
- `LoggingMiddleware` не печатает контент сообщения — приватность соблюдается.
- `.gitignore` покрывает `.env`, `logs/`, `__pycache__/`, `.venv/`, `.pytest_cache/`, IDE-каталоги.
- Тесты не делают сетевых вызовов (всё мокается), `pytest -q` проходит ~5 секунд.

## 5. Как добавлять новые записи

При обнаружении проблемы:

1. Найти подходящую секцию (§2 — баги/легаси, §3 — нюансы).
2. Создать подсекцию по шаблону:
   ```markdown
   ### 2.X Краткое название
   **Файл**: `app/...py:строка`. **Серьёзность**: низкая | средняя | высокая.

   Описание... Минимальное воспроизведение, если возможно...

   **Рекомендация**: что делать. См. `roadmap.md`, если запланировано.
   ```
3. Если решение запланировано — добавить запись в `_docs/roadmap.md`.
4. Если решение не запланировано — оставить «требует решения, кандидат на отдельную задачу».
5. **`_docs/legacy.md` обновляется только при добавлении нового подраздела `2.X`** — он указатель, а не справочник.

## 6. История закрытий

(Для будущих записей: когда баг исправлен — переносим запись сюда с указанием SHA коммита и даты.)

- **2026-04-26 — закрыт §2.1 «Нет команды `/reset` и нет `registry.reset()` для модели»** (спринт 03 «Conversation context», задача 4.1). Добавлены `UserSettingsRegistry.reset(user_id)` (`feat(services): add Summarizer for compressing dialog history via LLM` — в составе сопутствующих правок) и команда `/reset` в `app/handlers/commands.py::cmd_reset` (очистка `ConversationStore` + `UserSettingsRegistry.reset`), зарегистрирована в `app/main.py::set_my_commands` (`feat(handlers): add /reset command to clear history and reset per-user settings`).
