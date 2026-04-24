# План реализации проекта

Пошаговый план разработки Telegram-бота с локальной LLM (Ollama + aiogram) от пустого репозитория до готового MVP.

## Легенда статусов

- **ToDo** — задача готова к работе (Definition of Ready выполнен).
- **Progress** — задача в работе (взята разработчиком).
- **Done** — задача завершена, Definition of Done выполнен.
- **Blocked** — задача заблокирована (требуется внешнее действие или разрешение зависимости).

## Правила работы с планом

1. Задачи берутся строго сверху вниз: нельзя брать задачу, если предыдущая не `Done`.
2. В статусе `Progress` одновременно может находиться **только одна** задача.
3. При смене статуса обновляется поле `Статус` у задачи и фиксируется коммитом `chore(plan): ...`.
4. Подробный процесс выполнения одной задачи описан в `process.md`.

---

## Задача 1. Подготовка окружения и зависимостей

- **Статус:** Done
- **Этап roadmap:** 0
- **Зависит от:** —
- **Связанные документы:** `docs/instructions.md` §11, `docs/stack.md` §1–§3, §10, `docs/roadmap.md` «Этап 0».

### Описание

Подготовить локальное dev-окружение, необходимое для запуска проекта:

1. Установить Python 3.11+ (проверить `python -V`).
2. Установить Ollama (`https://ollama.com`), запустить сервис (`ollama serve`).
3. Загрузить модели из ТЗ (теги уточнить через `ollama list`):
   - `ollama pull qwen3.5:0.8b` (или аналогичный доступный тег семейства Qwen);
   - `ollama pull deepseek-r1:1.5b`.
4. Создать Telegram-бота через `@BotFather`, сохранить токен в надёжном месте (пока **не** коммитить).
5. Создать виртуальное окружение в корне репозитория: `python -m venv .venv`, активировать.

### Definition of Done

- `python -V` выводит версию ≥ 3.11.
- `ollama list` показывает обе модели (фактические теги зафиксированы в заметках — потребуются для `.env`).
- Токен Telegram получен и сохранён локально (вне репозитория).
- Каталог `.venv/` создан, активация работает, `pip --version` отрабатывает внутри окружения.
- В репозитории **не появилось** артефактов от установки (никаких `.env`, `.venv/` в `git status` не коммитится).

---

## Задача 2. Скелет репозитория и базовые файлы

- **Статус:** Done
- **Этап roadmap:** 1
- **Зависит от:** Задача 1
- **Связанные документы:** `docs/project-structure.md`, `docs/stack.md` §8–§9, `docs/roadmap.md` «Этап 1».

### Описание

Сформировать целевую структуру репозитория согласно `docs/project-structure.md`:

1. Создать `.gitignore` со всеми блоками из `docs/project-structure.md` §«Что должно попасть в `.gitignore`» (Python, Tests/tools, IDE, Logs, Secrets).
2. Создать `.env.example` с полным набором переменных из `docs/stack.md` §9 (`TELEGRAM_BOT_TOKEN`, `OLLAMA_BASE_URL`, `OLLAMA_DEFAULT_MODEL`, `OLLAMA_AVAILABLE_MODELS`, `OLLAMA_TIMEOUT`, `SYSTEM_PROMPT`, `LOG_LEVEL`, `LOG_FILE`) с комментариями-подсказками.
3. Создать `requirements.txt` минимум с:
   - `aiogram>=3.4,<4`
   - `ollama>=0.3`
   - `pydantic-settings>=2.1,<3`
   - `pytest>=8`
   - `pytest-asyncio>=0.23`
   - `pytest-mock>=3.12`
4. Создать структуру пакетов с `__init__.py`:
   - `app/`, `app/handlers/`, `app/services/`, `app/middlewares/`, `app/utils/`
   - `tests/`, `tests/services/`, `tests/handlers/`
5. Создать заготовку `README.md` в корне (структура разделов: «Установка», «Настройка», «Запуск», «Команды бота», «Тесты»; наполнение — на этапе задачи 10).
6. Удалить заглушку `main.py` из корня (функциональность переедет в `app/main.py`).

### Definition of Done

- Структура каталогов полностью соответствует `docs/project-structure.md` (включая тесты, зеркалирующие `app/`).
- `pip install -r requirements.txt` в активированном `.venv` проходит без ошибок.
- `git status` показывает только добавленные файлы, реальный `.env` отсутствует.
- `.env.example` содержит **все** 8 переменных, указанных в `docs/stack.md` §9.
- `python -c "import app"` не падает (пакет импортируется).

---

## Задача 3. Конфигурация (`Settings`) и логирование

- **Статус:** Done
- **Этап roadmap:** 2
- **Зависит от:** Задача 2
- **Связанные документы:** `docs/architecture.md` §3.2–§3.3, `docs/stack.md` §4–§5, `docs/instructions.md` §6–§7, `docs/testing.md` §3.1, `docs/roadmap.md` «Этап 2».

### Описание

1. Реализовать `app/config.py`:
   - Класс `Settings(BaseSettings)` на `pydantic-settings`.
   - Поля: `telegram_bot_token: SecretStr`, `ollama_base_url: str`, `ollama_default_model: str`, `ollama_available_models: list[str]` (парсинг из CSV), `ollama_timeout: int`, `system_prompt: str`, `log_level: str`, `log_file: str`.
   - `model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")`.
   - Валидатор: `ollama_default_model` должен присутствовать в `ollama_available_models`.
2. Реализовать `app/logging_config.py`:
   - Функция `setup_logging(settings: Settings) -> None`.
   - `logging.config.dictConfig` с двумя handler'ами:
     - `console` (`StreamHandler`, уровень из `settings.log_level`);
     - `file` (`RotatingFileHandler`, путь из `settings.log_file`, `maxBytes=1_000_000`, `backupCount=3`).
   - Формат: `%(asctime)s | %(levelname)s | %(name)s | %(message)s`.
   - Автосоздание каталога для файла логов (`Path(settings.log_file).parent.mkdir(parents=True, exist_ok=True)`).
   - Запрещено писать токен или другие секреты.
3. Тесты `tests/test_config.py`:
   - Корректная загрузка из `monkeypatch.setenv(...)` — все поля читаются.
   - `OLLAMA_AVAILABLE_MODELS=a,b,c` парсится в `["a","b","c"]`.
   - Отсутствие `TELEGRAM_BOT_TOKEN` → `ValidationError`.
   - `ollama_default_model` не в списке доступных → `ValidationError`.

### Definition of Done

- `pytest tests/test_config.py -q` зелёный, минимум 4 теста.
- `python -c "from app.config import Settings; print(Settings().ollama_default_model)"` при заполненном `.env` выводит имя модели и не падает.
- `setup_logging` не требует `app/` на диске: при запуске в tmp-каталоге создаёт файл логов по относительному пути.
- В логах при `LOG_LEVEL=DEBUG` не печатается значение `TELEGRAM_BOT_TOKEN` (убедиться ручной проверкой / тестом).
- Линтер/форматтер (если настроен) — зелёный.

---

## Задача 4. LLM-клиент (`OllamaClient`) и исключения

- **Статус:** Done
- **Этап roadmap:** 3
- **Зависит от:** Задача 3
- **Связанные документы:** `docs/architecture.md` §3.4, §5, `docs/stack.md` §3, `docs/instructions.md` §4–§5, `docs/testing.md` §3.2, `docs/roadmap.md` «Этап 3».

### Описание

1. В `app/services/llm.py`:
   - Иерархия исключений: `LLMError` (базовое) → `LLMTimeout`, `LLMUnavailable`, `LLMBadResponse` (например, 4xx/5xx, пустой ответ).
   - Класс `OllamaClient`:
     - Конструктор: `base_url: str`, `timeout: float`, опциональный внешний async http/ollama клиент (для тестов).
     - Метод `async generate(prompt: str, *, model: str, system_prompt: str | None) -> str`.
     - Метод `async list_models() -> list[str]` (опционально, можно отложить).
     - Метод `async close()` для graceful shutdown (закрыть HTTP-сессию).
   - Реализация через `ollama.AsyncClient` **или** `httpx.AsyncClient` (выбор зафиксировать в шапке файла docstring-ом).
   - Маппинг ошибок:
     - `httpx.ConnectError` / `ollama.ResponseError` connection → `LLMUnavailable`.
     - `httpx.TimeoutException` / `asyncio.TimeoutError` → `LLMTimeout`.
     - HTTP 404 («модель не найдена») → `LLMBadResponse` с сообщением «модель не найдена».
     - HTTP 5xx → `LLMBadResponse`.
   - Обязательное логирование каждого вызова: `user`-поле не здесь, но `model`, `len_in`, `len_out`, `dur_ms`, `status` — здесь.
2. Тесты `tests/services/test_llm_client.py`:
   - Успех: замоканный клиент → `generate` возвращает строку.
   - Timeout → `LLMTimeout`.
   - Connection error → `LLMUnavailable`.
   - HTTP 404 → `LLMBadResponse` с сообщением про модель.
   - HTTP 5xx → `LLMBadResponse`.
   - Пустой ответ модели → выбранная стратегия закреплена тестом.

### Definition of Done

- `pytest tests/services/test_llm_client.py -q` зелёный, минимум 6 тестов.
- `OllamaClient` не импортирует ничего из `aiogram` (проверить `grep`).
- Весь I/O — `async`, нет `time.sleep` / `requests`.
- Логирование работает: при успешном вызове в логе появляется строка с `model`, `len_in`, `len_out`, `dur_ms`.
- Исключения экспортируются из `app.services.llm` (есть в `__all__` или используются в хендлерах через явный импорт).

---

## Задача 5. Registry для модели и системного промпта

- **Статус:** Done
- **Этап roadmap:** 4
- **Зависит от:** Задача 4
- **Связанные документы:** `docs/architecture.md` §3.5, `docs/commands.md` §«/model», §«/prompt», `docs/testing.md` §3.3, `docs/roadmap.md` «Этап 4».

### Описание

1. В `app/services/model_registry.py`:
   - Класс `UserSettingsRegistry` (либо две разделённые: `ModelRegistry` + `PromptRegistry`) с методами:
     - `get_model(user_id: int) -> str` (возвращает default из конструктора, если ничего не установлено).
     - `set_model(user_id: int, model: str) -> None`.
     - `get_prompt(user_id: int) -> str`.
     - `set_prompt(user_id: int, prompt: str) -> None`.
     - `reset(user_id: int) -> None`.
   - Конструктор принимает `default_model: str`, `default_prompt: str`.
   - Потокобезопасность: `asyncio.Lock` (только если методы могут вызываться одновременно; иначе достаточно обычного dict — зафиксировать решение в docstring).
   - Состояние **только in-memory**, никаких файлов/БД (требование CON-1, CON-3).
2. Тесты `tests/services/test_model_registry.py`:
   - `get` без предварительного `set` → default.
   - `set` → последующий `get` возвращает заданное значение.
   - `reset` → снова default.
   - Изоляция между `user_id` (значение одного пользователя не влияет на другого).

### Definition of Done

- `pytest tests/services/test_model_registry.py -q` зелёный, минимум 4 теста.
- Модуль не содержит обращений к файловой системе / сети.
- Публичный API задокументирован docstring'ами (кратко).

---

## Задача 6. Entrypoint и команды `/start`, `/help`

- **Статус:** Done
- **Этап roadmap:** 5
- **Зависит от:** Задача 5
- **Связанные документы:** `docs/architecture.md` §3.1, §3.6, `docs/commands.md` §«/start», §«/help», §«BotFather / setMyCommands», `docs/stack.md` §2, `docs/roadmap.md` «Этап 5».

### Описание

1. `app/main.py`:
   - `async def main() -> None`:
     - Загрузить `Settings()`.
     - Вызвать `setup_logging(settings)`.
     - Создать `Bot(token=settings.telegram_bot_token.get_secret_value(), default=DefaultBotProperties(parse_mode=...))`.
     - Создать `Dispatcher()`.
     - Создать `llm_client = OllamaClient(...)`, `registry = UserSettingsRegistry(...)`.
     - Прокинуть в `dp["settings"]`, `dp["llm_client"]`, `dp["registry"]` (DI через `workflow_data`).
     - Подключить роутеры из `app/handlers/commands.py` (на этом этапе — только `/start`, `/help`).
     - Вызвать `await bot.set_my_commands([...])` со списком из `docs/commands.md` §«BotFather / setMyCommands».
     - Логировать «Bot started».
     - `await dp.start_polling(bot)` в `try/finally` с `await llm_client.close()` и `await bot.session.close()`.
2. `app/__main__.py`: `asyncio.run(main())`.
3. `app/handlers/commands.py`:
   - Роутер `router = Router(name="commands")`.
   - Хендлер `/start` — возвращает шаблон из `docs/commands.md` §«/start».
   - Хендлер `/help` — возвращает расширенную справку, включая текущую модель (`registry.get_model(user_id)`) и обрезанный до 200 символов текущий системный промпт.
4. Тесты:
   - `tests/handlers/test_commands.py`: `/start` → `message.answer` вызван с текстом, содержащим «Привет».
   - `/help` → содержит имя текущей модели.

### Definition of Done

- `python -m app` запускается, в консольном логе строка «Bot started».
- В Telegram команда `/start` возвращает приветствие со списком команд.
- В Telegram команда `/help` возвращает расширенную справку.
- `pytest tests/handlers/test_commands.py -q` зелёный, минимум 2 теста.
- Бот корректно завершается по Ctrl+C, без ошибок «Unclosed client session».
- В BotFather UI видны 5 команд после запуска.

---

## Задача 7. Основной обработчик текста → LLM

- **Статус:** Done
- **Этап roadmap:** 6
- **Зависит от:** Задача 6
- **Связанные документы:** `docs/architecture.md` §4–§5, `docs/commands.md` §«Произвольный текст», §«Ограничения ввода», `docs/testing.md` §3.4, `docs/roadmap.md` «Этап 6».

### Описание

1. `app/handlers/messages.py`:
   - Роутер `router = Router(name="messages")`.
   - Хендлер `F.text & ~F.text.startswith("/")`.
   - Алгоритм (см. `docs/commands.md` §«Произвольный текст»):
     1. `await message.bot.send_chat_action(chat_id, ChatAction.TYPING)`.
     2. `model = registry.get_model(user_id)`.
     3. `system = registry.get_prompt(user_id)`.
     4. Если `len(text) > 4000` — ответ «Слишком длинный запрос, сократите», выход.
     5. `response = await llm_client.generate(text, model=model, system_prompt=system)`.
     6. Успех → `message.answer(response)`; при `len > 4096` — разбивка через `app/utils/text.py::split_long_message`.
     7. `except LLMTimeout` → «Модель слишком долго отвечает.» + `logger.warning`.
     8. `except LLMUnavailable` → «LLM сейчас недоступна, попробуйте позже.» + `logger.error`.
     9. `except LLMBadResponse as e` → сообщение из исключения (или «Модель не найдена, выберите через /models»).
     10. `except LLMError` → общее «Произошла ошибка при обращении к LLM».
2. `app/utils/text.py::split_long_message(text: str, limit: int = 4096) -> list[str]` — разбивает по границам строк/пробелов.
3. Подключить `router` в `app/main.py`.
4. Тесты `tests/handlers/test_messages.py`:
   - Успех: мок `llm_client.generate` → `message.answer` вызван с ответом.
   - `LLMTimeout` → `answer` содержит «долго», `logger.warning` вызван.
   - `LLMUnavailable` → `answer` содержит «недоступна», `logger.error` вызван.
   - Длинный ответ (> 4096) → `answer` вызван несколько раз.
   - Слишком длинный ввод → `answer` с подсказкой, `generate` **не** вызывался.

### Definition of Done

- `pytest tests/handlers/ -q` зелёный, минимум 5 тестов для `test_messages.py`.
- В Telegram произвольный текст возвращает ответ от LLM (ручная проверка).
- При остановленной Ollama (`ollama stop` / `pkill ollama`) бот **не падает**, пользователь получает сообщение «LLM сейчас недоступна…», в лог пишется `ERROR`.
- Индикатор «печатает…» виден в чате во время генерации.
- В логе на каждое сообщение есть запись с `user`, `chat`, `model`, `dur_ms`.

---

## Задача 8. Команды `/models`, `/model <name>`, `/prompt [text]`

- **Статус:** Done
- **Этап roadmap:** 7
- **Зависит от:** Задача 7
- **Связанные документы:** `docs/commands.md` §«/models», §«/model», §«/prompt», `docs/testing.md` §3.4, `docs/roadmap.md` «Этап 7».

### Описание

1. В `app/handlers/commands.py` добавить хендлеры:
   - `/models`:
     - Вывести список из `settings.ollama_available_models`, отметив активную модель пользователя (`← активная`).
     - Подсказка по смене модели примером команды.
   - `/model <name>`:
     - Без аргумента → «Использование: `/model <имя>`, список: /models».
     - Имя не в `settings.ollama_available_models` → «Модель не найдена. Доступно: ...».
     - Иначе → `registry.set_model(user_id, name)` + ответ «Модель переключена на `<name>`.»
   - `/prompt [text]`:
     - Пустой аргумент → `registry.reset_prompt(user_id)` (или `set_prompt(user_id, settings.system_prompt)`) + «Системный промпт сброшен к значению по умолчанию.»
     - Непустой → `registry.set_prompt(user_id, text)` + «Системный промпт обновлён.»
2. Парсинг аргументов — через `command.args` (aiogram 3) или вручную через `message.text.split(maxsplit=1)`.
3. Тесты `tests/handlers/test_commands.py` (дополнительно):
   - `/models` → ответ содержит все модели из настроек и маркер активной.
   - `/model qwen3.5:0.8b` → `registry.set_model` вызван корректно; ответ «переключена».
   - `/model unknown` → `registry.set_model` **не** вызывался; ответ содержит «не найдена».
   - `/model` без аргумента → подсказка-использование.
   - `/prompt текст` → `registry.set_prompt` вызван; ответ «обновлён».
   - `/prompt` без аргумента → `registry.reset_prompt` (или `set_prompt` с дефолтом); ответ «сброшен».

### Definition of Done

- `pytest tests/handlers/test_commands.py -q` зелёный, минимум 6 доп. тестов.
- В Telegram все три команды работают, переключение модели видно в логе (`model=...`) и влияет на следующий запрос к LLM (ручная проверка).
- Никаких ошибок в консоли при невалидных аргументах (`/model `, `/model qwe rty`).

---

## Задача 9. Middleware логирования и глобальный error handler

- **Статус:** Done
- **Этап roadmap:** 8
- **Зависит от:** Задача 8
- **Связанные документы:** `docs/architecture.md` §3.7, §5, `docs/instructions.md` §5–§6, `docs/roadmap.md` «Этап 8».

### Описание

1. `app/middlewares/logging_mw.py`:
   - Класс `LoggingMiddleware(BaseMiddleware)`.
   - В `__call__`: засечь `t0`, вызвать `handler(event, data)`, залогировать `INFO`: `user=<id> chat=<id> type=<update_type> dur_ms=<n> status=<ok|error>`.
   - Не логировать контент сообщения (приватность).
   - Зарегистрировать на `dp.update.middleware(LoggingMiddleware())`.
2. `app/handlers/errors.py`:
   - Хендлер `@dp.errors`/`router.errors` (aiogram 3 API).
   - Логировать через `logger.exception` полный stacktrace.
   - Пытаться отправить пользователю нейтральное сообщение «Что-то пошло не так. Попробуйте ещё раз.» (если доступен `update.message`).
   - Не пробрасывать исключение дальше — polling должен продолжать работу.
3. В `app/main.py` зарегистрировать роутер `errors_router` и middleware.
4. Тесты (по возможности):
   - Для middleware — вызов с мок-handler'ом, проверка, что логгер вызван.
   - Для error handler — намеренно падающий хендлер → polling не падает (если сложно — покрыть ручной проверкой и зафиксировать в чек-листе MVP).

### Definition of Done

- Намеренно падающий хендлер (например, `raise RuntimeError("boom")` в dev-ветке) приводит к:
  - записи уровня `ERROR`/`EXCEPTION` в логе со stacktrace;
  - сообщению пользователю;
  - продолжению работы polling (следующее сообщение обрабатывается).
- На каждый апдейт в лог-файле есть ровно одна строка уровня `INFO` от middleware с полями `user`, `chat`, `type`, `dur_ms`, `status`.
- Содержимое сообщения пользователя в логах middleware отсутствует.
- `pytest -q` — зелёный (существующие тесты не сломаны).

---

## Задача 10. README, финальная полировка и чек-лист приёмки MVP

- **Статус:** Progress
- **Этап roadmap:** 9
- **Зависит от:** Задача 9
- **Связанные документы:** `docs/mvp.md` §5 «Критерии приёмки», `docs/instructions.md` §2, `docs/testing.md` §5–§6, `docs/roadmap.md` «Этап 9».

### Описание

1. Заполнить `README.md` в корне разделами:
   - **О проекте** (1 абзац).
   - **Требования** (Python 3.11+, Ollama, Telegram bot token).
   - **Установка** (клон, venv, `pip install -r requirements.txt`).
   - **Настройка** (скопировать `.env.example` в `.env`, описание каждой переменной).
   - **Запуск** (`ollama serve`, `python -m app`).
   - **Команды бота** (таблица из `docs/commands.md`).
   - **Тесты** (`pytest -q`, опционально покрытие).
   - **Ограничения MVP** (короткий список из `docs/mvp.md` §3).
2. Проверить `.gitignore`: `git status` не показывает `.env`, `logs/`, `.venv/`, `__pycache__/`.
3. Прогнать `pytest -q` — должно быть зелёно. Если установлен `pytest-cov`, прогнать `pytest --cov=app` и убедиться, что покрытие ≥ 70% (`services/` и `handlers/` ≥ 85%).
4. Прогнать чек-лист из `docs/mvp.md` §5 «Критерии приёмки» — все 10 пунктов отметить как выполненные (зафиксировать результаты в финальном коммите или в `progress.txt`).
5. Убедиться, что в истории коммитов нет реального токена (`git log -p | grep -i "TELEGRAM_BOT_TOKEN="` возвращает только примеры из `.env.example`).

### Definition of Done

- `README.md` содержит все 7 разделов выше, инструкция воспроизводима на чистой машине.
- `git status` чист после всех правок.
- `pytest -q` зелёный.
- Все 10 пунктов `docs/mvp.md` §5 отмечены как выполненные (ручной чек-лист с пометками пройден).
- В репозитории нет реального `.env` и нет утечек токена в истории.
- Финальный коммит тэгирован (опционально) как `mvp-ready`.

---

## Задача 11 (опционально). Доработки после MVP

- **Статус:** ToDo
- **Этап roadmap:** 10
- **Зависит от:** Задача 10
- **Связанные документы:** `docs/roadmap.md` «Этап 10», `docs/commands.md` §«/reset», `docs/stack.md` §11.

### Описание

Выполняется только после закрытия MVP и согласования с заказчиком. Любой из пунктов — отдельная мини-задача; можно разбить на подзадачи 11.1, 11.2, …:

1. Команда `/reset` — сброс модели и промпта пользователя (`registry.reset(user_id)`), тест.
2. Throttling middleware — простой rate-limit (например, 1 сообщение / 2 сек на пользователя), тест.
3. Dockerfile + `docker-compose.yml` (Ollama + бот), README-раздел «Docker».
4. GitHub Actions CI: `lint` (ruff) + `pytest`.
5. Streaming токенов из Ollama → редактирование исходящего сообщения чанками.

### Definition of Done

- Для каждой выбранной подзадачи: код + тесты + обновлённый `README.md`, `pytest -q` зелёный, ручная проверка.
- Документация `docs/` обновлена, если меняется поведение (`commands.md`, `architecture.md`, `stack.md`).

---

## Сводная таблица

| # | Задача                                           | Этап | Статус | Зависит от |
|---|--------------------------------------------------|:----:|:------:|:----------:|
| 1 | Подготовка окружения и зависимостей              | 0    | Done   | —          |
| 2 | Скелет репозитория и базовые файлы               | 1    | Done   | 1          |
| 3 | Конфигурация (`Settings`) и логирование          | 2    | Done   | 2          |
| 4 | LLM-клиент (`OllamaClient`) и исключения         | 3    | Done   | 3          |
| 5 | Registry для модели и системного промпта         | 4    | Done   | 4          |
| 6 | Entrypoint и команды `/start`, `/help`           | 5    | Done   | 5          |
| 7 | Основной обработчик текста → LLM                 | 6    | Done   | 6          |
| 8 | Команды `/models`, `/model`, `/prompt`           | 7    | Done   | 7          |
| 9 | Middleware логирования + глобальный error handler| 8    | Done   | 8          |
|10 | README, полировка, чек-лист приёмки MVP          | 9    | Progress | 9          |
|11 | (Опционально) Доработки после MVP                | 10   | ToDo   | 10         |
