# Внешние ссылки

Каталог ссылок на документацию используемых в проекте библиотек, моделей, протоколов и стандартов. Цель — не искать одно и то же повторно при работе над задачей. **Это не справочник по проекту**, а только указатели наружу.

При добавлении новой зависимости в `requirements.txt` или новой внешней системы — допишите сюда строку с краткой пометкой «зачем нужно в проекте» (модуль / функция / переменная окружения).

## 1. Telegram

| Что | Ссылка | Где в проекте |
|-----|--------|---------------|
| aiogram (3.x) | https://docs.aiogram.dev/ | `app/main.py`, `app/handlers/*` — `Bot`, `Dispatcher`, `Router`, `BaseMiddleware` |
| aiogram — Filters | https://docs.aiogram.dev/en/latest/dispatcher/filters/index.html | `app/handlers/messages.py` — `F.text & ~F.text.startswith("/")` |
| aiogram — Errors | https://docs.aiogram.dev/en/latest/dispatcher/errors.html | `app/handlers/errors.py` — `@router.errors(...)` |
| aiogram — `setMyCommands` | https://core.telegram.org/bots/api#setmycommands | `app/main.py::main` — список команд BotFather |
| Telegram Bot API | https://core.telegram.org/bots/api | общая спецификация (long polling, лимиты) |
| Telegram message limit (4096) | https://core.telegram.org/method/messages.sendMessage | `app/utils/text.py::TELEGRAM_MESSAGE_LIMIT` |
| BotFather | https://t.me/BotFather | получение `TELEGRAM_BOT_TOKEN` |

## 2. LLM-слой (Ollama)

| Что | Ссылка | Где в проекте |
|-----|--------|---------------|
| Ollama | https://ollama.com/ | LLM-рантайм, REST API на `localhost:11434`, см. `_docs/stack.md` §3 |
| Ollama REST API | https://github.com/ollama/ollama/blob/main/docs/api.md | базовый протокол (`/api/generate`) |
| Ollama Python SDK | https://github.com/ollama/ollama-python | `app/services/llm.py` — `ollama.AsyncClient`, `ollama.ResponseError` |
| Модель `qwen3.5:0.8b` | https://ollama.com/library/qwen | вариант `OLLAMA_DEFAULT_MODEL` |
| Модель `deepseek-r1:1.5b` | https://ollama.com/library/deepseek-r1 | альтернативная модель в `OLLAMA_AVAILABLE_MODELS` |
| httpx (исключения) | https://www.python-httpx.org/exceptions/ | `app/services/llm.py` — маппинг `TimeoutException`, `ConnectError` в `LLMError` |

## 3. Конфигурация и логирование

| Что | Ссылка | Где в проекте |
|-----|--------|---------------|
| pydantic | https://docs.pydantic.dev/latest/ | `field_validator`, `model_validator`, `SecretStr` в `app/config.py` |
| pydantic-settings | https://docs.pydantic.dev/latest/concepts/pydantic_settings/ | `app/config.py::Settings(BaseSettings)`, парсинг `.env` |
| Python `logging` | https://docs.python.org/3/library/logging.html | `app/logging_config.py`, логгеры во всех модулях |
| Python `logging.handlers` | https://docs.python.org/3/library/logging.handlers.html | `RotatingFileHandler` в `app/logging_config.py` |
| `logging.config.dictConfig` | https://docs.python.org/3/library/logging.config.html#logging.config.dictConfig | конфигурация логирования через словарь в `app/logging_config.py` |

## 4. Тестирование

| Что | Ссылка | Где в проекте |
|-----|--------|---------------|
| pytest | https://docs.pytest.org/ | основной test-runner, конфиг в `pyproject.toml` |
| pytest-asyncio | https://pytest-asyncio.readthedocs.io/ | `asyncio_mode = "auto"` в `pyproject.toml`, тесты `async def` |
| pytest-mock | https://pytest-mock.readthedocs.io/ | фикстура `mocker` в `tests/**/test_*.py` |
| `unittest.mock` | https://docs.python.org/3/library/unittest.mock.html | `MagicMock`, `AsyncMock` для мок-объектов `Message`/`Bot`/`OllamaClient` |

## 5. Стандарты разработки

| Что | Ссылка | Где в проекте |
|-----|--------|---------------|
| Conventional Commits | https://www.conventionalcommits.org/ | формат коммитов: `feat(scope): ...`, см. `_docs/instructions.md` §1 |
| Python typing | https://docs.python.org/3/library/typing.html | type hints обязательны в публичном API (см. `_docs/instructions.md` §2) |
| PEP 8 | https://peps.python.org/pep-0008/ | базовый стиль кода, `ruff` его реализует |
| `asyncio` | https://docs.python.org/3/library/asyncio.html | модель параллелизма для всего I/O |
| ruff | https://docs.astral.sh/ruff/ | целевой линтер/форматтер (опционально, не обязателен для MVP) |
