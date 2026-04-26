# ai_tg_bot

Telegram-бот, работающий как простой AI-чат на **локальной** LLM. Принимает текстовые сообщения, отправляет их в [Ollama](https://ollama.com), возвращает ответ. Помнит контекст диалога per-user (in-memory), сжимает длинные истории через LLM-суммаризацию. Никаких облачных API, никакой базы данных — история живёт только в памяти процесса и теряется при рестарте.

Построен на [`aiogram 3`](https://docs.aiogram.dev/) + `asyncio` + [`pydantic-settings`](https://docs.pydantic.dev/latest/concepts/pydantic_settings/), покрыт unit-тестами на `pytest` с моками (никаких сетевых вызовов в тестах).

## Возможности

- 🚀 **Long polling** через aiogram 3, без webhook.
- 🧠 **Локальная LLM** через Ollama (по умолчанию `qwen3.5:0.8b` и `deepseek-r1:1.5b`, набор настраивается через `.env`).
- 💬 **История диалога per-user** (in-memory): бот помнит предыдущие сообщения в рамках сессии. Лимит размера + автоудаление старейших + LLM-суммаризация при превышении порога. Подробнее — ниже.
- ⚙️ **Per-user настройки**: каждый пользователь может выбрать свою модель и системный промпт командой.
- 📑 **6 команд**: `/start`, `/help`, `/models`, `/model <name>`, `/prompt [<text>]`, `/reset`. Любой не-командный текст уходит в LLM с историей.
- 🛡 **Обработка ошибок LLM**: таймаут / недоступность / 4xx-5xx — каждое отдаётся пользователю человеческим сообщением, в лог пишется `WARNING`/`ERROR`. Падение суммаризации не валит ответ.
- 📜 **Логирование** в консоль и в файл с ротацией (`RotatingFileHandler`); метрики `user / chat / model / dur_ms / status` по каждому запросу. Перед каждым LLM-запросом — отдельная строка `llm_context messages=… tokens=…` (по умолчанию также с полным `payload=`). Контент пользовательских сообщений в обычные логи middleware не пишется.
- ✂️ **Разбивка длинных ответов** > 4096 символов на части (Telegram API limit).
- 🧪 **Unit-тесты** через моки: ~100 тестов проходят за секунды без реального Telegram/Ollama.

## Требования

- **Python 3.11+** (разрабатывалось на 3.12).
- **Ollama** (локальный сервис) с предзагруженными моделями.
- **Telegram bot token** от [@BotFather](https://t.me/BotFather).

## Установка

```bash
git clone https://github.com/radif-ru/ai_tg_bot.git
cd ai_tg_bot

python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

## Настройка

1. Скопировать шаблон и отредактировать секреты:

   ```bash
   cp .env.example .env
   # вписать TELEGRAM_BOT_TOKEN и при необходимости поменять модели / URL Ollama
   ```

2. Переменные окружения (полный список — в `_docs/stack.md` §9):

   | Переменная                  | Назначение                                                                 | Default |
   |-----------------------------|----------------------------------------------------------------------------|---------|
   | `TELEGRAM_BOT_TOKEN`        | Токен бота от @BotFather (обязательно).                                    | —       |
   | `OLLAMA_BASE_URL`           | URL локального Ollama API.                                                 | `http://localhost:11434` |
   | `OLLAMA_DEFAULT_MODEL`      | Модель по умолчанию (должна быть в `OLLAMA_AVAILABLE_MODELS`).             | —       |
   | `OLLAMA_AVAILABLE_MODELS`   | Список разрешённых моделей через запятую.                                  | —       |
   | `OLLAMA_TIMEOUT`            | Таймаут запроса к Ollama, секунды.                                         | `60`    |
   | `SYSTEM_PROMPT`             | Системный промпт по умолчанию (первое сообщение `role: system` в контексте). | `"Ты — полезный ассистент. Отвечай кратко и по делу."` |
   | `LOG_LEVEL`                 | Уровень логов: `DEBUG` \| `INFO` \| `WARNING` \| `ERROR`.                  | `INFO`  |
   | `LOG_FILE`                  | Путь к файлу логов (каталог создаётся автоматически).                      | `logs/bot.log` |
   | `HISTORY_MAX_MESSAGES`      | Жёсткий лимит сообщений в истории одного пользователя (FIFO-обрезка).      | `20`    |
   | `HISTORY_SUMMARY_THRESHOLD` | Порог: при достижении этого числа сообщений старая часть истории сжимается через LLM. Должно быть `> 0` и `<= HISTORY_MAX_MESSAGES`. | `10` |
   | `SUMMARIZATION_PROMPT`      | System prompt для LLM-вызова суммаризации.                                 | (см. `.env.example`) |
   | `LOG_LLM_CONTEXT`           | Логировать ли полный JSON-контекст перед каждым LLM-запросом. При `false` пишется только размер (`messages=`, `tokens=`). | `true` |

3. Загрузить модели в Ollama:

   ```bash
   ollama pull qwen3.5:0.8b
   ollama pull deepseek-r1:1.5b
   ollama list   # убедиться, что модели появились
   ```

## Запуск

```bash
ollama serve &              # если Ollama ещё не запущена
source .venv/bin/activate
python -m app
```

В консоли появится строка уровня `INFO`: `Bot started`. Остановить — `Ctrl+C`.

## Команды бота

| Команда                | Параметры       | Что делает                                                    |
|------------------------|-----------------|---------------------------------------------------------------|
| `/start`               | —               | Приветствие, краткая инструкция, список команд.               |
| `/help`                | —               | Подробная справка, текущая модель и системный промпт.         |
| `/models`              | —               | Список `OLLAMA_AVAILABLE_MODELS` с пометкой активной модели.  |
| `/model <name>`        | имя модели      | Переключить активную модель для текущего пользователя.        |
| `/prompt [<text>]`     | текст \| пусто  | Задать системный промпт; без аргумента — сброс к дефолту.     |
| `/reset`               | —               | Очистить историю диалога и сбросить per-user model/prompt.     |
| *произвольный текст*   | —               | Отправить в LLM с историей и получить ответ.                  |

Подробное поведение каждой команды — в `_docs/commands.md`.

## История диалога

Бот помнит контекст беседы **в рамках сессии процесса** — отдельно для каждого пользователя.

- **Где хранится:** в памяти процесса (`app/services/conversation.py::ConversationStore`). Никакой БД, файла или Redis. После перезапуска бота вся история теряется.
- **Формат сообщений:** список словарей `[{"role": "user"|"assistant"|"system", "content": "..."}, ...]` — тот же формат, что у `chat`-API Ollama. Перед каждым запросом в LLM собирается итоговый контекст: `[{"role": "system", "content": <SYSTEM_PROMPT>}] + история`.
- **Ограничение размера:** не больше `HISTORY_MAX_MESSAGES` сообщений (default — 20). При переполнении самые старые удаляются (FIFO). Резюме (`role: system`), которое появляется в результате суммаризации, тоже считается обычным сообщением для целей лимита.
- **Сброс:** командой `/reset` (или автоматически при рестарте процесса). `/reset` дополнительно сбрасывает per-user модель и системный промпт к default'ам из `.env`.
- **Логирование контекста:** перед каждым LLM-запросом в лог пишется строка вида `llm_context user=… chat=… model=… messages=N tokens=K payload=<JSON>`. Если `LOG_LLM_CONTEXT=false`, поле `payload=` опускается — остаются только размеры (полезно, если в логах не должно быть приватного текста). Оценка `tokens` — приближённая (`символы / 4`), точный токенайзер — кандидат на отдельный спринт.

## Суммаризация

Чтобы длинные диалоги не упирались в контекстное окно модели, бот сжимает старую часть истории через ту же LLM.

- **Когда срабатывает:** после успешного ответа модели длина истории сравнивается с `HISTORY_SUMMARY_THRESHOLD` (default — 10). При достижении порога запускается суммаризатор.
- **Как работает:** последние 2 сообщения (как правило, текущая пара user/assistant) сохраняются как есть. Всё, что было до них, отправляется в Ollama отдельным `chat`-вызовом с системным промптом из `SUMMARIZATION_PROMPT`. Полученный ответ записывается обратно в историю одним сообщением `{"role": "system", "content": "Краткое резюме предыдущей части диалога: …"}` — оно заменяет старую часть.
- **Что остаётся в истории после сжатия:** `[summary, last_user, last_assistant]` (3 сообщения). Дальнейший диалог продолжается уже с урезанным контекстом, и при следующем достижении порога процесс повторится.
- **Если суммаризация упала** (таймаут / недоступность Ollama) — это **не валит** ответ пользователю: пишется `WARNING summarize failed …`, история остаётся как есть и обрежется естественным FIFO при последующих сообщениях.
- **Качество резюме** зависит от выбранной локальной модели. `SUMMARIZATION_PROMPT` можно подкручивать через `.env` без правки кода.

## Тесты

```bash
pytest -q
```

С покрытием (если установлен `pytest-cov`):

```bash
pytest --cov=app --cov-report=term-missing
```

Тесты не делают сетевых вызовов — `aiogram.Bot`, `Message` и `ollama.AsyncClient` мокируются через `pytest-mock`.

## Структура проекта

```
app/                    # код приложения
├── main.py             # сборка Bot/Dispatcher и запуск polling
├── config.py           # Settings на pydantic-settings
├── logging_config.py   # dictConfig + RotatingFileHandler
├── handlers/           # /start, /help, /models, /model, /prompt + handler произвольного текста + errors
├── services/           # OllamaClient (LLM) + UserSettingsRegistry (per-user model/prompt)
├── middlewares/        # LoggingMiddleware
└── utils/              # split_long_message и пр.

app/services/
├── llm.py              # OllamaClient.generate / .chat + estimate_tokens
├── model_registry.py   # UserSettingsRegistry: per-user model + system prompt
├── conversation.py     # ConversationStore: in-memory история per-user
└── summarizer.py       # Summarizer: сжатие истории через LLM

tests/                  # зеркалит app/, моки aiogram и ollama
_docs/                  # проектная документация (см. _docs/README.md)
_board/                 # доска задач: индекс спринтов + sprints/<NN>-...
```

Полное дерево с описаниями — `_docs/project-structure.md`.

## Ограничения

По дизайну:

- **История диалога** живёт **только в памяти процесса** — после рестарта стирается. Персистентного хранилища (БД / файла) нет.
- Без **облачных LLM** — только локальный Ollama.
- Только **long polling**, без webhook.
- Игнорирует **нетекстовые сообщения** (фото, аудио, стикеры).
- Оценка токенов в логах — приближённая (`chars/4`), не точный токенайзер.

Известные нюансы кода (легаси / тех.долг) — `_docs/current-state.md` §2 и `_docs/legacy.md`.
Кандидаты в следующие спринты (Docker, CI, throttling, стриминг, точный токенайзер) — `_docs/roadmap.md` Этап 10.

## Документация

- 📘 [`_docs/README.md`](./_docs/README.md) — индекс проектной документации.
- 🏗️ [`_docs/architecture.md`](./_docs/architecture.md) — компоненты и поток данных.
- 💬 [`_docs/commands.md`](./_docs/commands.md) — спецификация команд бота.
- 📌 [`_docs/current-state.md`](./_docs/current-state.md) — фактическое состояние кода (читать перед правками).
- 🛠️ [`_docs/instructions.md`](./_docs/instructions.md) — правила разработки.
- 🔗 [`_docs/links.md`](./_docs/links.md) — каталог внешних ссылок (aiogram, Ollama, pydantic, …).
- 📋 [`_board/README.md`](./_board/README.md) — процесс работы со спринтами и задачами.
