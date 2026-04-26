# Архитектура

## 1. Общая схема

Pipeline с in-memory историей диалога per-user и LLM-суммаризацией, без БД и без персистентного хранилища:

```
                +---------------------+
                |   Telegram User     |
                +----------+----------+
                           |
                           | текстовое сообщение
                           v
                +---------------------+
                |   Telegram Bot API  |
                +----------+----------+
                           |
                           | long polling
                           v
                +---------------------+         +-----------------------+
                |   aiogram Bot       |<------> | ConversationStore     |
                |   (Dispatcher,      |  read/  | (in-memory per-user   |
                |    Router, Handlers)|  write  |  history)             |
                +----------+----------+         +-----------------------+
                           |                                ^
                           | async-вызов                    | summary
                           v                                |
                +---------------------+         +-----------------------+
                |  LLM Service Layer  | <------ | Summarizer            |
                | OllamaClient        |  chat() | (сжимает историю   |
                | .generate / .chat   |         |  через LLM)          |
                +----------+----------+         +-----------------------+
                           |
                           | HTTP (ollama REST API: /api/generate, /api/chat)
                           v
                +---------------------+
                |   Ollama (local)    |
                |  qwen3.5:0.8b /     |
                |  deepseek-r1:1.5b   |
                +---------------------+
```

Обратный путь: ответ LLM → сервис-слой → handler → допись в `ConversationStore` → (опционально) `Summarizer.summarize` → `bot.send_message` → Telegram → пользователь.

## 2. Принципы

- **In-memory state, no persistence**: пер-сессионное состояние (история, выбранная модель, системный промпт) живёт только в памяти процесса и теряется при рестарте. БД, файлы, Redis и другие персистентные хранилища не используются.
- **Per-user изоляция**: история и настройки разделены по `user_id`; один пользователь не видит контекст другого.
- **Async-first**: весь код на `async/await`, I/O не блокирует event loop.
- **Polling, не webhook**: `bot.start_polling()` / `Dispatcher.start_polling()`.
- **Отказоустойчивость**: любая ошибка (таймаут LLM, сетевой сбой, недоступность Ollama, сбой суммаризации) ловится и превращается в понятное сообщение пользователю + запись в лог.
- **Конфигурация через env**: все секреты и настройки — из переменных окружения (`.env`).

## 3. Компоненты

### 3.1 Точка входа (`app/main.py` / `bot.py`)
- Загружает конфигурацию (`Settings`).
- Инициализирует `Bot`, `Dispatcher`, логгер.
- Регистрирует роутеры (handlers).
- Запускает polling.

### 3.2 Конфигурация (`app/config.py`)
- Класс `Settings` на базе `pydantic-settings` (или `os.environ` + dataclass).
- Параметры:
  - `TELEGRAM_BOT_TOKEN` — токен бота.
  - `OLLAMA_BASE_URL` — URL Ollama (по умолчанию `http://localhost:11434`).
  - `OLLAMA_DEFAULT_MODEL` — модель по умолчанию.
  - `OLLAMA_AVAILABLE_MODELS` — список разрешённых моделей.
  - `OLLAMA_TIMEOUT` — таймаут запроса.
  - `SYSTEM_PROMPT` — системный промпт по умолчанию (первое сообщение `role: system` в каждом контексте).
  - `LOG_LEVEL`, `LOG_FILE` — параметры логирования.
  - `HISTORY_MAX_MESSAGES` — жёсткий лимит размера истории на пользователя.
  - `HISTORY_SUMMARY_THRESHOLD` — порог запуска суммаризации (`> 0`, `<= HISTORY_MAX_MESSAGES`).
  - `SUMMARIZATION_PROMPT` — system prompt для LLM-вызова суммаризации.
  - `LOG_LLM_CONTEXT` — печатать ли полный JSON-контекст перед каждым LLM-запросом (см. §4 п.5).

### 3.3 Логирование (`app/logging_config.py`)
- `logging.config.dictConfig` с двумя handler'ами: консоль и файл (ротация).
- Формат: timestamp, level, logger, message.
- Логируются: старт бота, входящие команды, входящий текст, запрос в LLM, длительность и статус, ошибки.

### 3.4 LLM-сервис (`app/services/llm.py`)
- Класс `OllamaClient` (async, на `ollama.AsyncClient`).
- Два пути вызова:
  - `generate(prompt: str, *, model, system_prompt) -> str` — одношаговый вызов (`/api/generate`); используется для обратной совместимости.
  - `chat(messages: list[dict], *, model) -> str` — полный контекст в формате `[{"role", "content"}, …]` (`/api/chat`); основной путь в handler'е текста.
- Маппинг ошибок (идентичен для обоих методов): `httpx.TimeoutException` / `asyncio.TimeoutError` → `LLMTimeout`; `httpx.ConnectError` → `LLMUnavailable`; `ollama.ResponseError` 404 → `LLMBadResponse("модель не найдена")`; прочие 4xx/5xx → `LLMBadResponse`; пустой текст → `LLMBadResponse("LLM вернула пустой ответ")`.
- На каждый вызов пишет INFO-строку с метриками (`model`, `len_in`, `len_out`, `dur_ms`, `status`; для `chat` — дополнительно `messages=N`).
- Функция уровня модуля `estimate_tokens(value: str | list[dict]) -> int` — грубая оценка `символы / 4`. Не точный токенайзер (для кириллицы реальное число может отличаться в 1.5–2 раза), но хватает для логирования размера контекста и порогов. Точный токенайзер (`tiktoken` / HF) — кандидат `roadmap.md`.

### 3.5 Per-user runtime-состояние

In-memory состояние процесса, разделённое по `user_id`. После рестарта всё возвращается к default'ам из `Settings`.

- **`UserSettingsRegistry` (`app/services/model_registry.py`)** — `user_id → model` + `user_id → system_prompt`. АПИ: `get_model` / `set_model` / `reset_model`, `get_prompt` / `set_prompt` / `reset_prompt`, `reset` (оба сразу).
- **`ConversationStore` (`app/services/conversation.py`)** — `user_id → list[{role, content}]`, история диалога. АПИ: `get_history`, `add_user_message`, `add_assistant_message`, `replace_with_summary(summary, kept_tail=2)`, `clear`. Жёсткий лимит `Settings.history_max_messages` (default 20); при переполнении самые старые сообщения удаляются (FIFO). `get_history` возвращает копию (не внутренний список) — мутации снаружи не влияют на стор.

Оба класса — без локов: в CPython в одном event loop'е dict-ассайны атомарны, а данные изолированы по `user_id`. Никакого файлового/сетевого I/O в сторах нет — они чистые in-memory структуры.

### 3.6 Handlers (`app/handlers/`)
- `commands.py`: `/start`, `/help`, `/model`, `/models`, `/prompt`.
- `messages.py`: обработчик произвольного текста → вызов LLM → ответ.
- `errors.py`: глобальный error handler на уровне Dispatcher.

### 3.7 Middleware (опционально, `app/middlewares/`)
- `LoggingMiddleware` — логирует каждый апдейт (user_id, chat_id, тип, длительность).
- `ThrottlingMiddleware` — простой rate-limit (опционально, за рамками MVP).

### 3.8 Суммаризация диалога (`app/services/summarizer.py`)

Класс `Summarizer` — тонкая обёртка над `OllamaClient.chat`, которая отправляет старую часть истории в LLM с системным промптом из `Settings.summarization_prompt` и возвращает краткое резюме.

- Конструктор: `Summarizer(client: OllamaClient, *, prompt: str)`. Промпт инъектируется параметром (без прямой зависимости от `Settings`), чтобы упростить юнит-тесты и позволить переопределение на ходу.
- `async summarize(messages, *, model)` — формирует payload `[{"role": "system", "content": prompt}] + messages` и вызывает `client.chat(payload, model=model)`.
- Суммаризатор не глушит ошибки и не ретраит — любая `LLMError` пробрасывается наверх; handler текста ловит её и пишет `WARNING summarize failed …`, не портя базовый ответ пользователю.

Порог запуска и политика «что оставляем как есть» — в хендлере текста (`app/handlers/messages.py`): срабатывает при `len(history) >= Settings.history_summary_threshold` (default 10), последние `kept_tail = 2` сообщения сохраняются как есть, остальное заменяется одним `{"role": "system", "content": "Краткое резюме предыдущей части диалога: …"}` через `ConversationStore.replace_with_summary`.

## 4. Поток обработки текстового сообщения

1. aiogram получает `Message` через polling.
2. `LoggingMiddleware` логирует входящий апдейт (`user/chat/type/dur_ms/status` — без контента).
3. Router направляет в `messages.py:handle_text`.
4. Handler:
   1. Проверяет длину ввода (`MAX_INPUT_LENGTH = 4000`); при превышении — подсказка пользователю и выход.
   2. Берёт `model` и `system_prompt` из `UserSettingsRegistry`.
   3. Дописывает сообщение пользователя в `ConversationStore`.
   4. Собирает контекст `messages = [{"role": "system", "content": system_prompt}] + history`.
   5. **Обязательно логирует контекст** перед запросом: `INFO llm_context user=… chat=… model=… messages=N tokens=K [payload=<JSON>]`. Поле `payload=` пишется только при `Settings.log_llm_context=True`.
   6. Показывает `bot.send_chat_action(ChatAction.TYPING)`, вызывает `OllamaClient.chat(messages, model=...)`.
   7. При успехе — дописывает ответ ассистента в `ConversationStore`.
   8. **Условная суммаризация**: если `len(history) >= Settings.history_summary_threshold`, вызывает `Summarizer.summarize(history[:-2], model=model)`, результат пишет в стор через `replace_with_summary(…, kept_tail=2)`. Падение суммаризации ловится в `LLMError` → `WARNING`, история остаётся как есть.
   9. Ответ пользователю — `message.answer(response)` (с разбивкой по границе 4096).
  10. При ошибке основного LLM-вызова — человеческое сообщение по таблице §5 + запись в лог.

## 5. Обработка ошибок

| Сценарий                          | Действие                                                        |
|-----------------------------------|-----------------------------------------------------------------|
| Ollama недоступна (connection)    | Лог ERROR, сообщение «LLM сейчас недоступна, попробуйте позже». |
| Таймаут запроса                   | Лог WARNING, сообщение «Модель слишком долго отвечает».         |
| Неизвестная модель / 404          | Лог ERROR, сообщение «Модель не найдена, выберите через /models».|
| Пустой / слишком длинный ввод     | Сообщение-подсказка пользователю.                               |
| Необработанное исключение handler | Перехват в глобальном `errors.py`, лог, нейтральный ответ.      |

## 6. Конкурентность и производительность

- aiogram + `asyncio` позволяет обрабатывать несколько апдейтов конкурентно.
- HTTP-клиент к Ollama — один на приложение (shared `AsyncClient`).
- Ollama сама сериализует запросы к модели (узкое место — GPU/CPU), но бот не блокируется: пока одна задача ждёт LLM, другие апдейты продолжают диспетчеризоваться.

## 7. Расширяемость

- Добавление новых моделей — правка `OLLAMA_AVAILABLE_MODELS`.
- Переключение на webhook — заменить запуск Dispatcher'а, не меняя сервис-слой.
- Персистентность истории — добавить адаптер `ConversationStore` поверх БД/Redis (сейчас — только in-memory).
- Точный токенайзер (`tiktoken` / HF-tokenizers) вместо эвристики `chars/4` в `estimate_tokens`.
- Throttling, стриминг ответа, Docker/CI — см. `roadmap.md` Этап 10.
