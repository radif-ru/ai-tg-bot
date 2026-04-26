# Инструкции по разработке

Этот документ описывает правила, которым должен следовать разработчик (и/или LLM-ассистент) при работе над проектом.

## 1. Git-дисциплина

- Основная ветка — `main` (или `master`), рабочие ветки — feature-branches.
- Коммиты — атомарные, сообщения в императиве: `feat: add /model command`, `fix: handle ollama timeout`, `test: cover llm client errors`.
- `.gitignore` обязательно содержит: `.env`, `venv/`, `.venv/`, `__pycache__/`, `*.pyc`, `logs/`, `.pytest_cache/`, `.mypy_cache/`, `.ruff_cache/`, `.idea/`, `.vscode/`.
- Секреты никогда не коммитить. Если токен случайно попал — ротировать в `@BotFather`, удалить из истории (`git filter-repo`).

## 2. Стиль кода

- Python 3.11+, type hints **обязательны** в сервис-слое и публичных функциях.
- Форматирование — `ruff format` (или `black`), длина строки 100.
- Импорты — `ruff` / `isort`-совместимо: stdlib → third-party → local.
- Имена: `snake_case` для функций/переменных, `PascalCase` для классов, `UPPER_CASE` для констант.
- Docstrings — краткие, на русском или английском, единообразно в рамках файла.
- Никаких `print` в проде — только `logging`.

## 3. Async-дисциплина

- Любой I/O (HTTP, файлы в hot path, Telegram API) — только через `await`.
- Не использовать `requests`, `time.sleep`, блокирующие SDK. Только `httpx.AsyncClient`, `aiofiles` (если нужно), `ollama.AsyncClient`.
- Не создавать новый event loop внутри handlers. Всё работает в loop'е, запущенном aiogram.
- Общие клиенты (HTTP, Ollama) — создаются **один раз на приложение**, закрываются при shutdown.

## 4. Обработка ошибок

- Каждый handler либо сам ловит ожидаемые исключения (`LLMError`, `asyncio.TimeoutError`), либо полагается на глобальный error handler в Dispatcher.
- Необработанных исключений быть не должно (см. `requirements.md` § нефункциональные).
- Сообщения пользователю — человеческие, без stacktrace. Stacktrace — только в лог (`logger.exception(...)`).
- Custom-исключение `LLMError` объявляется в `app/services/llm.py` и используется во всём LLM-слое.

## 5. Логирование

- Конфиг один — `app/logging_config.py`.
- Каждый запрос логируется хотя бы одной записью INFO: `user=<id> chat=<id> model=<name> len_in=<n> len_out=<n> dur_ms=<n>`.
- Ошибки — `logger.exception` или `logger.error(..., exc_info=True)`.
- Чувствительные данные (токен) — никогда в логи.

## 6. Секреты и конфиг

- Все секреты и настройки — из `.env` через `pydantic-settings`.
- `.env` — в `.gitignore`, в репо — `.env.example` с описанием каждого поля.
- Не хардкодить токены, URL'ы, имена моделей в коде. Всё через `Settings`.

## 7. Тестирование

См. подробно `testing.md`. Минимум:
- `pytest` + `pytest-asyncio`.
- Unit-тесты для `OllamaClient` (мок HTTP).
- Unit-тесты для handler'ов (мок `OllamaClient`).
- `pytest` должен проходить локально одной командой без реального Telegram/Ollama.

**Обязательное правило (тесты на новое поведение):** новый или изменённый код в `app/` не принимается без unit-теста на новое поведение. Целевое покрытие из `testing.md` §5: пакет `app/` ≥ 70%, `app/services/` и `app/handlers/` ≥ 85%. Чисто-документационные задачи (правят только `_docs/`, `_board/`, `README.md`, `.env.example`) от этого правила освобождены — в DoD таких задач явно ставится `n/a` напротив пункта про тесты.

**Обязательное правило (зелёный pytest перед коммитом):** перед каждым коммитом задачи прогнать `pytest -q` локально. Коммит идёт **только при зелёном результате**. Если тесты падают (даже не относящиеся к задаче) — сначала чинятся, и только потом фиксируются изменения. Это касается любых коммитов задачи, кроме `chore(plan): start/complete task ...` (они не меняют код).

## 8. Процесс добавления фичи

1. Описать поведение в `requirements.md` или `commands.md`.
2. При необходимости обновить `architecture.md` (новый компонент / поток).
3. Написать/обновить тест (красный).
4. Реализовать (зелёный).
5. Отрефакторить, прогнать линтер.
6. Обновить README, если появилась новая команда/параметр.
7. **Прогнать `pytest -q` — все тесты зелёные.** Без этого шага коммит не делается (см. §7 «Обязательное правило (зелёный pytest перед коммитом)»).
8. Коммит + push.

## 9. Что НЕЛЬЗЯ делать

- Добавлять БД, хранение истории, persistent state — **противоречит stateless-архитектуре** (`requirements.md` CON-1, CON-3).
- Использовать облачные LLM (OpenAI, Anthropic, Google и т. п.) — проект построен вокруг **локальной** LLM (`requirements.md` CON-2).
- Переходить на webhook в MVP — получение апдейтов только через long polling (`requirements.md` CON-4).
- Коммитить `.env`, реальный токен, логи.
- Писать синхронный I/O в event loop'е.

## 10. Локальный запуск (dev)

```bash
# 1) окружение
python -m venv .venv
source .venv/bin/activate  # Windows/WSL: source .venv/bin/activate
pip install -r requirements.txt

# 2) секреты
cp .env.example .env
# отредактировать .env — вписать TELEGRAM_BOT_TOKEN

# 3) Ollama
ollama serve &   # если ещё не запущен
ollama pull qwen3.5:0.8b
ollama pull deepseek-r1:1.5b

# 4) запуск бота
python -m app

# 5) тесты
pytest -q
```
