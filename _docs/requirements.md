# Требования

Документ формализует требования к проекту в виде проверяемых FR / NFR / CON / ASM. Источник — постановка задачи на Telegram-бота с локальной LLM, изложенная в `mvp.md` и `architecture.md`.

## 1. Функциональные требования (FR)

| ID    | Требование |
|-------|------------|
| FR-1  | Бот принимает текстовые сообщения от пользователя в Telegram. |
| FR-2  | Каждое текстовое сообщение передаётся в локальную LLM и ответ возвращается пользователю. |
| FR-3  | Каждый пользователь имеет независимую in-memory историю диалога. История ограничена по размеру и при превышении порога суммаризируется. Персистентного хранения нет — история теряется при рестарте процесса. |
| FR-4  | Поддерживается команда `/start` с приветственным сообщением. |
| FR-5  | Поддерживается системный промпт, настраиваемый через конфигурацию/команду. |
| FR-6  | Поддерживается выбор LLM-модели через команду бота. |
| FR-7  | Поддерживаются как минимум модели `qwen3.5:0.8b` и `deepseek-r1:1.5b` через Ollama (конкретные теги — по доступности в окружении). |
| FR-8  | Получение апдейтов — через polling, **не webhook**. |
| FR-9  | Все запросы к LLM логируются (user, chat, модель, длительность, статус). |
| FR-10 | Секреты загружаются из `.env` (через переменные окружения). |
| FR-11 | В репозитории есть `.gitignore`, который исключает `.env`, логи, кэши, venv. |
| FR-12 | В репозитории есть `README.md` с инструкцией запуска и описанием команд. |
| FR-13 | Перед каждым LLM-запросом полный контекст (`messages`) и его размер (в приближённых токенах) логируются. Печать полного payload управляется флагом `LOG_LLM_CONTEXT`. |
| FR-14 | История ограничена по количеству сообщений (`HISTORY_MAX_MESSAGES`); при превышении самые старые сообщения удаляются (FIFO). |
| FR-15 | При достижении `HISTORY_SUMMARY_THRESHOLD` сообщений старая часть истории сжимается LLM в краткое резюме и заменяет соответствующие сообщения в истории; при падении суммаризации ответ пользователю всё равно выдаётся. |
| FR-16 | Команда `/reset` стирает историю пользователя и сбрасывает per-user модель и системный промпт к default'ам из `Settings`. |

## 2. Нефункциональные требования (NFR)

| ID     | Требование |
|--------|------------|
| NFR-1  | Стек: **Python** + **aiogram**. |
| NFR-2  | Весь код I/O — асинхронный (`async/await`). |
| NFR-3  | Нет необработанных исключений: все ошибки логируются и пользователю возвращается понятное сообщение. |
| NFR-4  | Код покрыт автоматическими тестами (`pytest`). |
| NFR-5  | Нет БД, нет persistent-хранилища. |
| NFR-6  | Нет использования облачных LLM. |
| NFR-7  | Бот стабильно обрабатывает несколько последовательных/параллельных запросов. |
| NFR-8  | Логи пишутся в консоль и в файл (с ротацией); файл логов — в `.gitignore`. |
| NFR-9  | Архитектура — минимальная: Telegram → Bot → LLM → Bot → Telegram. Без избыточных слоёв. |

## 3. Ограничения (CON)

| ID    | Ограничение |
|-------|-------------|
| CON-1 | Запрещено **персистентное** хранение истории диалога (БД, файл, Redis). In-memory история, теряющаяся при рестарте процесса, разрешена (реализует FR-3). |
| CON-2 | Запрещены облачные LLM. |
| CON-3 | Запрещена БД. |
| CON-4 | Запрещён webhook (только polling). |

## 4. Предположения (ASM)

| ID    | Предположение |
|-------|---------------|
| ASM-1 | У разработчика локально установлен и запущен Ollama на стандартном порту. |
| ASM-2 | Модели, указанные в `.env`, заранее скачаны в Ollama (`ollama pull <model>`). |
| ASM-3 | Telegram-бот создан через `@BotFather` и токен доступен. |
| ASM-4 | Per-user runtime-состояние (выбранная модель, системный промпт, история диалога, суммаризированный контекст) живёт только в памяти процесса. После рестарта всё возвращается к default'ам из `Settings`. |
| ASM-5 | Используется Python 3.11+. |

## 5. Трассировка требований → компоненты

| Требование | Модуль / артефакт |
|------------|-------------------|
| FR-1, FR-2 | `app/handlers/messages.py`, `app/services/llm.py` |
| FR-3       | `app/services/conversation.py::ConversationStore` (in-memory per-user) + вызов `OllamaClient.chat(messages, ...)` в `app/handlers/messages.py`. |
| FR-4       | `app/handlers/commands.py::cmd_start` |
| FR-5       | `Settings.SYSTEM_PROMPT` + `/prompt` в `commands.py`; system prompt выставляется первым сообщением контекста в `app/handlers/messages.py`. |
| FR-6       | `/model`, `/models` в `commands.py` + `app/services/model_registry.py::UserSettingsRegistry` |
| FR-7       | `Settings.OLLAMA_AVAILABLE_MODELS`, `.env.example` |
| FR-8       | `app/main.py::main` → `dp.start_polling(bot)` |
| FR-9       | `app/middlewares/logging_mw.py` и `OllamaClient.generate / .chat` |
| FR-10, FR-11 | `app/config.py`, `.env.example`, `.gitignore` |
| FR-12      | `README.md` в корне |
| FR-13      | `app/handlers/messages.py::_log_context` + `app/services/llm.py::estimate_tokens`; флаг `Settings.log_llm_context`. |
| FR-14      | `app/services/conversation.py::ConversationStore._truncate` + `Settings.history_max_messages`. |
| FR-15      | `app/services/summarizer.py::Summarizer.summarize` + `app/handlers/messages.py::_maybe_summarize` + `ConversationStore.replace_with_summary` + `Settings.history_summary_threshold` / `summarization_prompt`. |
| FR-16      | `app/handlers/commands.py::cmd_reset` (вызывает `ConversationStore.clear` + `UserSettingsRegistry.reset`); регистрация в `app/main.py::set_my_commands`. |
| NFR-1..9   | Общие правила в `instructions.md` |

## 6. Критерии верификации

Каждое требование должно быть подтверждено как минимум одним из способов:
- **T** — автоматический тест (`pytest`).
- **M** — ручная проверка по сценарию.
- **I** — инспекция кода/репозитория (например, наличие `.env.example`, `.gitignore`).

| ID     | Способ |
|--------|--------|
| FR-1   | T + M  |
| FR-2   | T + M  |
| FR-3   | T (тесты `ConversationStore`, тесты handler'а «второе сообщение видит предыдущую пару») + M |
| FR-4   | T + M  |
| FR-5   | T + M  |
| FR-6   | T + M  |
| FR-7   | M (проверка в реальном Ollama) |
| FR-8   | I (код использует polling) |
| FR-9   | T (проверка записи в лог) + I |
| FR-10  | I      |
| FR-11  | I      |
| FR-12  | I      |
| FR-13  | T (`test_context_log_with_payload_when_log_llm_context_true` / `..._false`) + I |
| FR-14  | T (`test_truncate_drops_oldest_when_over_limit`) |
| FR-15  | T (`test_summarizer_called_when_history_reaches_threshold`, `test_summarizer_failure_does_not_break_response`) + M |
| FR-16  | T (`test_reset_clears_history_and_resets_registry`, `test_reset_works_on_real_store_and_registry`) + M |
| NFR-1  | I      |
| NFR-2  | I + T  |
| NFR-3  | T + M  |
| NFR-4  | I (наличие тестов) + прогон `pytest` |
| NFR-5  | I      |
| NFR-6  | I      |
| NFR-7  | M (нагрузочный сценарий на 3–5 параллельных чатов) |
| NFR-8  | M + I  |
| NFR-9  | I      |
