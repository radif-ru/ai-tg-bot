# Структура проекта

Целевая структура репозитория после реализации MVP.

```
ai_tg_bot/
├── .env.example              # шаблон конфигурации (коммитится)
├── .env                      # реальные секреты (в .gitignore)
├── .gitignore
├── CLAUDE.md                 # поведенческие гайдлайны LLM-агента (общие)
├── README.md                 # инструкция запуска + команды бота
├── requirements.txt          # runtime + dev-зависимости
├── pyproject.toml            # конфиг pytest (asyncio_mode=auto)
│
├── _docs/                     # проектная документация для LLM-агентов (этот каталог)
│   ├── README.md             # индекс документации
│   ├── mvp.md                # scope MVP и критерии приёмки
│   ├── requirements.md       # FR / NFR / CON / ASM
│   ├── architecture.md       # компоненты, потоки данных, обработка ошибок
│   ├── stack.md              # версии, зависимости, переменные окружения
│   ├── instructions.md       # правила разработки (проектные)
│   ├── project-structure.md  # этот файл
│   ├── commands.md           # спецификация команд бота
│   ├── testing.md            # стратегия тестирования
│   ├── roadmap.md            # этапы развития и кандидаты в спринты
│   ├── current-state.md      # фактическое состояние: легаси, баги, нюансы
│   ├── legacy.md             # указатель на тех.долг (ссылки на current-state.md и roadmap.md)
│   └── links.md              # каталог внешних ссылок (aiogram, Ollama, pydantic, …)
│
├── _board/                    # процесс и итерации для LLM-агентов
│   ├── README.md             # индекс доски
│   ├── plan.md               # индекс спринтов + шаблоны (сами задачи — в sprints/)
│   ├── process.md            # пошаговый процесс выполнения одной задачи
│   ├── progress.txt          # ad-hoc заметки о прогрессе / чек-лист приёмки
│   └── sprints/              # файлы спринтов: <NN>-<short-name>.md (ТЗ + этапы + задачи)
│
├── logs/                      # файлы логов (в .gitignore)
│   └── bot.log
│
├── app/                       # код приложения
│   ├── __init__.py
│   ├── __main__.py            # entrypoint: python -m app
│   ├── main.py                # async def main(): запуск polling
│   ├── config.py              # Settings на pydantic-settings
│   ├── logging_config.py      # dictConfig для logging
│   │
│   ├── handlers/
│   │   ├── __init__.py
│   │   ├── commands.py        # /start, /help, /models, /model, /prompt
│   │   ├── messages.py        # F.text & ~F.text.startswith('/') → LLM
│   │   └── errors.py          # глобальный error handler (router.errors)
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── llm.py             # OllamaClient (.generate, .chat) + estimate_tokens + LLMError/LLMTimeout/LLMUnavailable/LLMBadResponse
│   │   ├── model_registry.py  # UserSettingsRegistry: user_id → model + prompt
│   │   ├── conversation.py    # ConversationStore: in-memory история диалога per-user
│   │   └── summarizer.py      # Summarizer: сжатие старой части истории через LLM
│   │
│   ├── middlewares/
│   │   ├── __init__.py
│   │   └── logging_mw.py      # LoggingMiddleware (user/chat/type/dur_ms/status)
│   │
│   └── utils/
│       ├── __init__.py
│       └── text.py            # split_long_message
│
└── tests/                     # зеркалирует app/
    ├── __init__.py
    ├── conftest.py            # фикстура base_env (изоляция переменных окружения)
    ├── test_config.py
    ├── test_logging_config.py
    ├── test_main.py           # smoke-тест сборки main()
    ├── test_middleware_logging.py
    ├── test_utils_text.py
    ├── services/
    │   ├── __init__.py
    │   ├── test_llm_client.py
    │   ├── test_model_registry.py
    │   ├── test_conversation_store.py
    │   └── test_summarizer.py
    └── handlers/
        ├── __init__.py
        ├── test_commands.py
        ├── test_errors.py
        └── test_messages.py
```

## Назначение ключевых модулей

| Путь | Ответственность |
|------|-----------------|
| `CLAUDE.md` | Общие поведенческие гайдлайны LLM-агента (think before coding, simplicity, surgical changes, goal-driven). Читается первым перед любой задачей. |
| `_docs/*.md` | Проектная документация: требования, архитектура, стек, инструкции, roadmap. |
| `_docs/current-state.md` | Фактическое состояние кода: что работает, легаси, известные проблемы. |
| `_docs/legacy.md` | Сводный указатель на технический долг (ссылки на `current-state.md` §2 и этапы `roadmap.md`). |
| `_docs/links.md` | Каталог внешних ссылок (aiogram, Ollama, pydantic, pytest и др.). |
| `_board/plan.md` | Индекс спринтов: легенда статусов, правила, шаблоны. Сами задачи — в `sprints/<NN>-<short-name>.md`. |
| `_board/sprints/` | Каталог файлов спринтов (по одному на спринт); история сохраняется. |
| `_board/process.md` | Пошаговый алгоритм выполнения одной задачи. |
| `_board/progress.txt` | Ad-hoc заметки о прогрессе / фиксация чек-листа приёмки. |
| `app/__main__.py` | Запуск `asyncio.run(main())`. |
| `app/main.py` | Собирает `Bot`, `Dispatcher`, регистрирует роутеры и middleware, стартует polling. |
| `app/config.py` | Класс `Settings(BaseSettings)`, парсинг `.env`, валидация. |
| `app/logging_config.py` | Функция `setup_logging(settings)` → `dictConfig`. |
| `app/handlers/commands.py` | Router с обработчиками команд (`/start`, `/help`, `/models`, `/model`, `/prompt`, `/reset`). |
| `app/handlers/messages.py` | Router с обработчиком `F.text & ~F.text.startswith('/')` → история → LLM `chat` → история → (опц.) суммаризация. |
| `app/handlers/errors.py` | `@router.errors()` — единая точка для необработанных ошибок. |
| `app/services/llm.py` | `OllamaClient` (async) с методами `generate` и `chat(messages, ...)`, утилитой `estimate_tokens`, иерархией `LLMError` → `LLMTimeout`/`LLMUnavailable`/`LLMBadResponse`. |
| `app/services/model_registry.py` | `UserSettingsRegistry`: in-memory `user_id → model` и `user_id → prompt`. |
| `app/services/conversation.py` | `ConversationStore`: in-memory история диалога per-user (`user_id → list[{role, content}]`), FIFO-обрезка, `replace_with_summary`, `clear`. |
| `app/services/summarizer.py` | `Summarizer`: тонкая обёртка над `OllamaClient.chat` для сжатия старой части истории в краткое резюме. |
| `app/middlewares/logging_mw.py` | Логирование каждого апдейта (`user`, `chat`, `type`, `dur_ms`, `status`). |
| `app/utils/text.py` | `split_long_message` — разбивка длинных ответов LLM по границам строк/пробелов. |
| `tests/` | Зеркалирует `app/`, unit-тесты с моками; никаких сетевых вызовов. |

## Принципы организации

- **Слои не протекают**: handler не знает про HTTP, сервис не знает про aiogram. `ConversationStore` и `Summarizer` тоже не зависят ни от aiogram, ни от ollama (только от `OllamaClient` в случае `Summarizer`).
- **Интерфейсы тонкие**: handler вызывает `await llm_client.chat(messages, model=...)` — и всё.
- **DI через aiogram `workflow_data`**: `dp["llm_client"]`, `dp["registry"]`, `dp["conversation"]`, `dp["summarizer"]`, `dp["settings"]`; handler получает их через параметры (aiogram 3 умеет инжектить по имени).
- **Тесты рядом с тем, что тестируют**: `tests/services/` зеркалит `app/services/`.

## Что должно попасть в `.gitignore`

```gitignore
# Python
__pycache__/
*.py[cod]
*.egg-info/
.venv/
venv/

# Tests / tools
.pytest_cache/
.mypy_cache/
.ruff_cache/
.coverage
htmlcov/

# IDE
.idea/
.vscode/

# Logs
logs/
*.log

# Secrets
.env
.env.*
!.env.example
```
