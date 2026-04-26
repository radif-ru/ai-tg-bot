# Спринт 03. Conversation Context

- **Источник:** ТЗ пользователя (2026-04-26): «реализовать поддержку контекста диалога — историю сообщений per-user, ограничение размера контекста, суммаризацию длинных диалогов, system prompt как часть контекста, обязательное логирование контекста перед каждым запросом в LLM».
- **Ветка:** `feature/conversation-context` (от `main`).
- **Открыт:** 2026-04-26
- **Закрыт:** —

## 1. Цель спринта

Снять MVP-ограничение «stateless» (FR-3 / CON-1) и сделать бота диалоговым: каждый пользователь получает свою in-memory историю сообщений в формате `[{role, content}, …]`, которая **целиком** отправляется в LLM при каждом ответе. История ограничивается по размеру (по умолчанию — последние 20 сообщений), а при превышении настраиваемого порога старая часть **сжимается LLM в краткое резюме** и заменяется одним system-сообщением. System prompt становится первым элементом контекста; перед каждым запросом в LLM полный контекст и его размер (в приблизительных токенах) логируются.

После закрытия спринта бот:

- помнит предыдущие сообщения в коротком диалоге;
- не «зависает» и не теряется в длинном (благодаря суммаризации);
- даёт **разные** ответы при наличии и отсутствии контекста (проверяемо через `/reset`).

## 2. Скоуп и non-goals

- **В скоупе:**
  - Per-user in-memory история диалога (`app/services/conversation.py::ConversationStore`).
  - Ограничение размера истории (по количеству сообщений), автоудаление самых старых при превышении.
  - Суммаризация старой части истории через LLM при превышении настраиваемого порога (`app/services/summarizer.py::Summarizer`).
  - System prompt как первое сообщение контекста (`role: system`).
  - Метод `OllamaClient.chat(messages, model)` поверх `ollama.AsyncClient.chat` (для отправки полной истории + system); `generate(...)` остаётся для обратной совместимости.
  - Логирование полного контекста и его размера в токенах перед каждым LLM-запросом (управляется флагом конфигурации).
  - Команда `/reset` — стирает историю + сбрасывает per-user модель и промпт; регистрация в `set_my_commands`; обновление `/start`/`/help`.
  - Расширение `Settings` и `.env.example` новыми параметрами (`HISTORY_MAX_MESSAGES`, `HISTORY_SUMMARY_THRESHOLD`, `SUMMARIZATION_PROMPT`, `LOG_LLM_CONTEXT`).
  - Тесты на каждый новый сервис, обновление существующих тестов handler'а текста.
  - Обновление документации: `README.md`, `_docs/architecture.md`, `_docs/requirements.md`, `_docs/commands.md`, `_docs/current-state.md`, `_docs/project-structure.md`, `.env.example`.
- **Вне скоупа (non-goals):**
  - Персистентность истории (БД, файл, Redis) — история по-прежнему живёт только в памяти процесса и теряется при рестарте.
  - Точный токенайзер (`tiktoken` / Hugging Face). Используем простую оценку «символы / 4», достаточно для логирования и порогов.
  - Throttling, стриминг ответов, Docker, CI — кандидаты следующих спринтов (`_docs/roadmap.md` Этап 10).
  - YAML-сериализация истории для долговременного хранения (опциональный пункт ТЗ; не делаем — формат логирования контекста решается в задаче 3.1).
  - Изменение поведения существующих команд `/models`, `/model`, `/prompt` (только `/reset` добавляется и `/start`/`/help` обновляются по тексту).

## 3. Acceptance Criteria спринта

- [ ] В `app/services/conversation.py` есть `ConversationStore` с per-user in-memory историей и автоматическим удалением старых сообщений при превышении `history_max_messages`.
- [ ] В `app/services/llm.py` появился `OllamaClient.chat(messages, *, model)` с теми же гарантиями маппинга ошибок, что и `generate(...)`; `generate(...)` не сломан (старые тесты зелёные).
- [ ] В `app/services/summarizer.py` есть `Summarizer.summarize(messages)`, использующий тот же `OllamaClient` и настраиваемый промпт суммаризации.
- [ ] Handler `app/handlers/messages.py` собирает контекст `[system] + history`, пишет его в лог вместе с количеством токенов **перед** запросом в LLM, после ответа дописывает реплики в историю и при превышении `history_summary_threshold` запускает суммаризацию.
- [ ] Команда `/reset` стирает историю пользователя и сбрасывает model+prompt; зарегистрирована в `set_my_commands`; упомянута в текстах `/start` и `/help`.
- [ ] `Settings` принимает `HISTORY_MAX_MESSAGES`, `HISTORY_SUMMARY_THRESHOLD`, `SUMMARIZATION_PROMPT`, `LOG_LLM_CONTEXT`; `.env.example` содержит все четыре переменные с комментариями.
- [ ] `pytest -q` зелёный; добавлены: тесты `ConversationStore`, тесты `Summarizer`, тесты `OllamaClient.chat`, обновлённые тесты `tests/handlers/test_messages.py` (контекст пробрасывается, история обновляется, при превышении порога вызывается суммаризатор), тест команды `/reset`.
- [ ] В `README.md` есть разделы «История диалога» (как реализовано хранение) и «Суммаризация» (как и когда срабатывает); упомянута команда `/reset`.
- [ ] В `_docs/architecture.md`, `_docs/requirements.md`, `_docs/commands.md`, `_docs/current-state.md`, `_docs/project-structure.md` обновлены формулировки про stateless / отсутствие истории; FR-3 переформулирован, CON-1 уточнён («запрещена персистентная история», но in-memory разрешена); добавлены FR на историю/лимит/суммаризацию/логирование контекста.
- [ ] Ручная проверка в Telegram: бот помнит «как меня зовут» в коротком диалоге; после `/reset` — забывает; в длинном диалоге (≥ `history_summary_threshold` сообщений) лог содержит запись о суммаризации и контекст не растёт неограниченно.
- [ ] Все задачи спринта — `Done`, сводная таблица актуальна.

## 4. Этап 1. Конфигурация и хранилище истории

Базовая инфраструктура: новые параметры в `Settings`, in-memory `ConversationStore` per-user. Этот этап ничего не отправляет в LLM — только структуры и тесты.

### Задача 1.1. Расширить `Settings` и `.env.example` параметрами истории

- **Статус:** Done
- **Приоритет:** high
- **Объём:** S
- **Зависит от:** —
- **Связанные документы:** `_docs/instructions.md` § «Секреты и конфиг», `_docs/stack.md` §9, `_docs/requirements.md` (раздел будет дополнен в задаче 5.3).
- **Затрагиваемые файлы:** `app/config.py`, `.env.example`, `tests/test_config.py`.

#### Описание

Добавить в `Settings` (см. `app/config.py`) четыре новых поля с разумными default'ами:

1. `history_max_messages: int = 20` — жёсткий лимит на количество сообщений в истории одного пользователя (без учёта system-prompt и summary). При превышении — удаляются самые старые.
2. `history_summary_threshold: int = 10` — порог количества сообщений в истории, при достижении которого запускается суммаризация (ТЗ упоминает «больше 5» как ориентир — берём 10 как практичный default; настраиваемо через `.env`). Должно быть `< history_max_messages`.
3. `summarization_prompt: str = "Кратко и точно резюмируй ключевые факты и решения из этого диалога в 2–4 предложениях. Ответ — только текст резюме, без вступлений."` — system prompt для LLM-вызова суммаризации.
4. `log_llm_context: bool = True` — печатать ли полный контекст в лог перед каждым LLM-запросом. Если `False` — печатается только размер контекста (количество сообщений + приблизительные токены).

Валидация (`@model_validator(mode="after")`): `history_summary_threshold <= history_max_messages` (иначе суммаризация никогда не сработает); оба значения > 0.

В `.env.example` — добавить все четыре переменные с комментариями-подсказками в раздел `# --- Conversation context ---`.

Тесты (`tests/test_config.py`):

1. Default-значения присутствуют (без переопределения в `.env`).
2. CSV-наследие не ломается (старые тесты зелёные).
3. `HISTORY_SUMMARY_THRESHOLD > HISTORY_MAX_MESSAGES` → `ValidationError`.
4. `HISTORY_MAX_MESSAGES = 0` → `ValidationError`.

#### Definition of Done

- [x] `Settings` принимает все 4 новых поля; default'ы соответствуют описанию.
- [x] Валидатор отбрасывает `threshold > max` и непозитивные значения.
- [x] `.env.example` содержит секцию `# --- Conversation context ---` со всеми 4 переменными и комментариями.
- [x] `pytest tests/test_config.py -q` зелёный, добавлены тесты под новые валидации.
- [x] `python -c "from app.config import Settings"` не падает (синтаксис ОК).

---

### Задача 1.2. `ConversationStore` — in-memory история per-user

- **Статус:** Progress
- **Приоритет:** high
- **Объём:** M
- **Зависит от:** Задача 1.1
- **Связанные документы:** `_docs/architecture.md` §3.5 (будет дополнено в задаче 5.2), `_docs/instructions.md` § «Async и I/O» (никаких блокировок).
- **Затрагиваемые файлы:** `app/services/conversation.py` (новый), `tests/services/test_conversation_store.py` (новый).

#### Описание

Создать `app/services/conversation.py` с классом `ConversationStore`:

1. Конструктор: `max_messages: int` (берётся из `Settings.history_max_messages`).
2. Внутреннее состояние — `dict[int, list[dict[str, str]]]` (`user_id -> [{"role": ..., "content": ...}, …]`); ничего, кроме `role` и `content`, не хранить.
3. Публичные методы:
   - `get_history(user_id) -> list[dict[str, str]]` — возвращает **копию** списка, чтобы handler не мог случайно мутировать внутреннее состояние.
   - `add_user_message(user_id, content)` — добавляет `{"role": "user", "content": content}` и сразу вызывает `_truncate(user_id)`.
   - `add_assistant_message(user_id, content)` — аналогично с `role="assistant"`.
   - `replace_with_summary(user_id, summary, *, kept_tail)` — заменяет первые `len(history) - kept_tail` сообщений на одно `{"role": "system", "content": <summary с префиксом>"}` (префикс — `"Краткое резюме предыдущей части диалога: "`); последние `kept_tail` сообщений сохраняются без изменений. После замены вновь вызывается `_truncate(user_id)`. Если `kept_tail >= len(history)` — no-op.
   - `clear(user_id)` — удаляет запись пользователя.
   - `__len__`-style helper не нужен; для измерений использовать `len(get_history(user_id))`.
4. Внутренний `_truncate(user_id)`: если длина истории > `max_messages` — обрезает с **начала** до `max_messages` (FIFO). Резюме (`role: system`) считается обычным сообщением для целей лимита.
5. Никаких файловых/сетевых вызовов, никаких локов (см. `app/services/model_registry.py` — тот же подход, GIL + изоляция по `user_id`).
6. Все методы — синхронные (это in-memory dict, async не нужен).

Тесты (`tests/services/test_conversation_store.py`, ≥ 6 сценариев):

1. `get_history(unknown_user)` → пустой список.
2. Добавление user/assistant в правильном порядке.
3. Превышение `max_messages` → старейшие удалены, длина = `max_messages`.
4. Изоляция историй между двумя `user_id`.
5. `clear(user_id)` → следующий `get_history` снова пустой.
6. `replace_with_summary`: проверить, что первые `N - kept_tail` сообщений заменены одним system-сообщением с заданным префиксом, последние `kept_tail` — без изменений; общая длина = `1 + kept_tail`.
7. `get_history` возвращает копию (мутация результата не влияет на стор).

#### Definition of Done

- [ ] Файл `app/services/conversation.py` создан, содержит только `ConversationStore` (никаких лишних утилит).
- [ ] Все 6 публичных операций реализованы по спецификации; внутреннее состояние не утекает наружу.
- [ ] `pytest tests/services/test_conversation_store.py -q` зелёный, ≥ 6 тестов.
- [ ] `grep -E "(import aiogram|from aiogram|import ollama|from ollama)" app/services/conversation.py` пусто (стор не зависит ни от Telegram, ни от LLM).

---

## 5. Этап 2. LLM-клиент: chat-API и суммаризатор

В этом этапе LLM-слой получает возможность принимать список сообщений целиком (`role/content`), а не только `prompt + system_prompt`, и отдельный сервис умеет вызывать LLM для сжатия диалога.

### Задача 2.1. `OllamaClient.chat(messages, model)` поверх Ollama chat-API

- **Статус:** ToDo
- **Приоритет:** high
- **Объём:** M
- **Зависит от:** Задача 1.2
- **Связанные документы:** `_docs/architecture.md` §3.4 (будет дополнено в задаче 5.2), `_docs/links.md` § «Ollama» (REST `/api/chat`), `_docs/testing.md` §3.2, существующая реализация `OllamaClient.generate`.
- **Затрагиваемые файлы:** `app/services/llm.py`, `tests/services/test_llm_client.py`.

#### Описание

Добавить в `app/services/llm.py`:

1. Метод `async chat(self, messages: list[dict[str, str]], *, model: str) -> str`:
   - Проксирует в `self._client.chat(model=model, messages=messages, stream=False)`.
   - Возвращает `resp.message.content` (или `resp["message"]["content"]` — что отдаёт `ollama.AsyncClient.chat`).
   - Маппинг ошибок — **тот же**, что в `generate()`: `httpx.TimeoutException` / `asyncio.TimeoutError` → `LLMTimeout`; `httpx.ConnectError` → `LLMUnavailable`; `ollama.ResponseError` 404 → `LLMBadResponse("модель не найдена")`; прочие 4xx/5xx → `LLMBadResponse`; пустой ответ → `LLMBadResponse("LLM вернула пустой ответ")`.
   - Логирование INFO с метриками: `model`, `len_in` (суммарная длина `content` всех сообщений), `len_out`, `dur_ms`, `status`, `messages_count`.
2. Утилита-функция `estimate_tokens(text_or_messages)` (на уровне модуля или статическим методом):
   - Принимает строку или `list[dict]`; возвращает `int` — приблизительное количество токенов.
   - Эвристика: `max(1, ceil(total_chars / 4))` (типичное соотношение для латиницы; для кириллицы — пессимистичнее, но это всё равно лог-метрика, не лимит на стороне модели).
   - Документировано в docstring как «грубая оценка, не точный токенайзер».
3. `generate(...)` оставить как есть (его используют существующие тесты и, возможно, суммаризация — задача 2.2 решит, переходить ли на `chat` или оставить `generate`).

Не дублировать код маппинга ошибок: вынести `try/except`-блок в приватный async-helper `_call(awaitable, *, model, prompt_len)` или продублировать аккуратно — на ваше усмотрение, главное чтобы оба метода не разъезжались по поведению.

Тесты (`tests/services/test_llm_client.py`, ≥ 5 новых):

1. `chat` success → возвращает `message.content`, в логе строка с `messages_count=N` и `status=ok`.
2. `chat` timeout (`httpx.TimeoutException`) → `LLMTimeout`.
3. `chat` connect error → `LLMUnavailable`.
4. `chat` 404 → `LLMBadResponse("не найдена")`.
5. `chat` пустой content → `LLMBadResponse`.
6. `estimate_tokens("hello world")` → ожидаемое значение (например, 3); `estimate_tokens([{"role":"user","content":"abc"},{"role":"assistant","content":"defg"}])` — сумма по `content`.

#### Definition of Done

- [ ] `OllamaClient.chat(messages, model=...)` реализован, мапит ошибки идентично `generate()`.
- [ ] `estimate_tokens` доступна как функция уровня модуля или статический метод (то, что удобнее импортировать в handler).
- [ ] `pytest tests/services/test_llm_client.py -q` зелёный; старые тесты `generate(...)` не сломаны; добавлено ≥ 5 новых.
- [ ] `grep -E "(import aiogram|from aiogram)" app/services/llm.py` пусто (LLM-слой остаётся изолированным).

---

### Задача 2.2. `Summarizer` — сжатие старой части диалога через LLM

- **Статус:** ToDo
- **Приоритет:** high
- **Объём:** S
- **Зависит от:** Задача 2.1
- **Связанные документы:** `_docs/architecture.md` §3.4 (будет дополнено в задаче 5.2), `Settings.summarization_prompt` (Задача 1.1).
- **Затрагиваемые файлы:** `app/services/summarizer.py` (новый), `tests/services/test_summarizer.py` (новый).

#### Описание

Создать `app/services/summarizer.py` с классом `Summarizer`:

1. Конструктор: `client: OllamaClient`, `prompt: str` (берётся из `settings.summarization_prompt`).
2. Метод `async summarize(self, messages: list[dict[str, str]], *, model: str) -> str`:
   - Формирует payload: `[{"role": "system", "content": self._prompt}] + messages`.
   - Вызывает `await self._client.chat(payload, model=model)`.
   - Возвращает текст ответа без модификаций (handler сам обернёт его в формат «Краткое резюме …»).
3. Никаких retry / fallback — если LLM упала, исключение пробрасывается наверх (handler обработает по той же таблице, что и обычные `LLMError`).

Тесты (`tests/services/test_summarizer.py`, ≥ 3 сценария):

1. `summarize(...)` вызывает `client.chat` с правильным payload (system-сообщение с заданным промптом + переданная история) и моделью.
2. Возвращает то, что вернул `client.chat`.
3. Если `client.chat` бросает `LLMTimeout` — оно пробрасывается дальше (Summarizer не глушит).

#### Definition of Done

- [ ] Файл `app/services/summarizer.py` создан, содержит только `Summarizer`.
- [ ] `pytest tests/services/test_summarizer.py -q` зелёный, ≥ 3 теста.
- [ ] Класс не зависит от `aiogram` и от `Settings` напрямую (получает `prompt` параметром).

---

## 6. Этап 3. Интеграция в handler текста

Здесь отдельные сервисы соединяются в один пайплайн: сообщение → история → контекст → лог → LLM → ответ → история → (опционально) суммаризация.

### Задача 3.1. Handler текста: контекст, логирование, LLM.chat, обновление истории

- **Статус:** ToDo
- **Приоритет:** high
- **Объём:** L
- **Зависит от:** Задача 1.2, 2.1, 2.2
- **Связанные документы:** `_docs/architecture.md` §4 (будет дополнено в задаче 5.2), `_docs/commands.md` § «Произвольный текст» (будет обновлено в задаче 5.4), `_docs/instructions.md` § «Логирование», ТЗ §5 «Логирование контекста».
- **Затрагиваемые файлы:** `app/handlers/messages.py`, `app/main.py`, `tests/handlers/test_messages.py`.

#### Описание

1. В `app/main.py`:
   - Создать `ConversationStore(max_messages=settings.history_max_messages)`.
   - Создать `Summarizer(client=llm_client, prompt=settings.summarization_prompt)`.
   - Прокинуть оба объекта в `dispatcher["conversation"]` и `dispatcher["summarizer"]`.

2. В `app/handlers/messages.py::handle_text` заменить текущий путь «один `generate(...)` без контекста» на следующий:

   1. Базовые проверки длины ввода — без изменений.
   2. `model = registry.get_model(user_id)`, `system_prompt = registry.get_prompt(user_id)`.
   3. `conversation.add_user_message(user_id, text)`.
   4. `history = conversation.get_history(user_id)`; `messages = [{"role": "system", "content": system_prompt}] + history`.
   5. **Логирование контекста перед LLM-запросом** (до `await llm_client.chat(...)`):
      - Если `settings.log_llm_context` is `True`: `logger.info("llm_context user=%s chat=%s model=%s messages=%d tokens=%d payload=%s", …, json.dumps(messages, ensure_ascii=False))`.
      - Если `False`: то же, но без `payload=...` (только размеры).
      - `tokens` — через `estimate_tokens(messages)` (Задача 2.1).
   6. `response = await llm_client.chat(messages, model=model)`.
   7. `conversation.add_assistant_message(user_id, response)`.
   8. **Суммаризация**: если `len(conversation.get_history(user_id)) >= settings.history_summary_threshold`:
      - Берём `kept_tail = 2` (последние user+assistant); `to_summarize = history[:-kept_tail]` (где `history` — свежий `get_history`).
      - Вызов `summary = await summarizer.summarize(to_summarize, model=model)`.
      - `conversation.replace_with_summary(user_id, summary, kept_tail=kept_tail)`.
      - В лог — INFO «summarized history»: `user`, `len_before`, `len_after`, `dur_ms`.
      - Любая `LLMError` в суммаризации **не должна** валить handler: ловим, пишем WARNING, оставляем историю как есть (она урежется естественным FIFO в `_truncate`).
   9. `await message.answer(response)` (с разбивкой длинных).

3. Обработка `LLMError` в основном `chat`-вызове остаётся такой же, как раньше (timeout / unavailable / bad response / общий LLMError → понятные сообщения). При ошибке LLM **последнее user-сообщение из истории убирать не надо** — пусть остаётся, чтобы пользователь мог переслать его повторно по смыслу (при желании можно явно откатить — решение принимается в коде, главное единообразно и с тестом).

4. Тесты (`tests/handlers/test_messages.py`, ≥ 4 новых):

   - `handle_text` вызывает `llm_client.chat` с `messages = [system] + history`, где `history` содержит сообщение пользователя (через мок-store + ассерты на аргументы вызова).
   - После успешного ответа история содержит и user, и assistant сообщения.
   - При длине истории `>= history_summary_threshold` вызывается `summarizer.summarize`, и `replace_with_summary` срабатывает (через мок Summarizer + ассерт).
   - Перед `chat`-вызовом в caplog появляется строка `llm_context user=… messages=… tokens=…` (с `payload=...` при `log_llm_context=True`, без — при `False`).
   - При `LLMError` от `summarizer.summarize` handler **не падает** и **возвращает основной ответ** пользователю (граница: упала только суммаризация, основной chat прошёл успешно).

#### Definition of Done

- [ ] В `app/main.py` создаются и прокидываются `ConversationStore` и `Summarizer`.
- [ ] `handle_text` использует `llm_client.chat(messages, …)` с `messages = [system] + history`; никаких прямых вызовов `generate(...)` в handler'е текста не остаётся.
- [ ] В лог перед каждым LLM-запросом пишется строка с `messages=N` и `tokens=K`; при `LOG_LLM_CONTEXT=True` — также `payload=<JSON>`.
- [ ] Суммаризация запускается при достижении порога; ошибка суммаризации не валит ответ пользователю (есть тест).
- [ ] `pytest tests/handlers/ -q` зелёный, ≥ 4 новых теста в `test_messages.py`; старые сценарии (timeout, unavailable, bad response, LLMError) — обновлены под новый API и зелёные.
- [ ] Ручная проверка в Telegram: «Меня зовут Радиф. Запомни.» → следующая реплика «Как меня зовут?» — бот отвечает корректно. После `/reset` (см. задачу 4.1) — забывает.

---

## 7. Этап 4. Команда `/reset`

Минимальная UX-обвязка для «забыть всё»: одна команда стирает историю + сбрасывает per-user model и prompt.

### Задача 4.1. Команда `/reset` и обновление справочных текстов

- **Статус:** ToDo
- **Приоритет:** medium
- **Объём:** S
- **Зависит от:** Задача 3.1
- **Связанные документы:** `_docs/commands.md` § «/reset» (будет уточнено в задаче 5.4), `_docs/current-state.md` §2.1 (после спринта запись будет закрыта).
- **Затрагиваемые файлы:** `app/handlers/commands.py`, `app/main.py`, `tests/handlers/test_commands.py`.

#### Описание

1. В `app/handlers/commands.py` добавить:
   ```python
   @router.message(Command("reset"))
   async def cmd_reset(
       message: Message,
       registry: UserSettingsRegistry,
       conversation: ConversationStore,
   ) -> None: ...
   ```
   - Берёт `user_id`, вызывает `conversation.clear(user_id)` и `registry.reset(user_id)` (общий reset уже существует).
   - Отвечает: «Контекст диалога очищен, модель и системный промпт сброшены к значениям по умолчанию.»

2. В `app/main.py` добавить `BotCommand(command="reset", description="Очистить контекст и сбросить настройки")` в `set_my_commands(...)` (порядок — после `/prompt`, перед или после — на ваше усмотрение, но команда должна попасть в UI).

3. Обновить тексты `START_TEXT` (`/start`) и `cmd_help` (`/help`) — добавить строку:
   ```
   /reset — очистить контекст и сбросить настройки
   ```

4. Тесты (`tests/handlers/test_commands.py`, ≥ 2 новых):

   - `/reset` вызывает `conversation.clear(user_id)` и `registry.reset(user_id)` (через моки).
   - `/reset` отвечает текстом, содержащим «Контекст диалога очищен» (или эквивалентный).

#### Definition of Done

- [ ] `/reset` зарегистрирован, корректно делает оба сброса.
- [ ] В UI Telegram (BotFather menu) команда видна (ручная проверка).
- [ ] Тексты `/start` и `/help` упоминают `/reset`.
- [ ] `pytest tests/handlers/test_commands.py -q` зелёный, добавлены ≥ 2 теста на `/reset`.

---

## 8. Этап 5. Документация и приёмка

Финальный этап: README, `_docs/`, `.env.example`, ручная приёмка по AC спринта. Этап делается **последним**, после задач 1–4 в `Done`, чтобы документация отражала фактический код.

### Задача 5.1. Обновить `README.md`: разделы «История диалога» и «Суммаризация»

- **Статус:** ToDo
- **Приоритет:** high
- **Объём:** S
- **Зависит от:** Задача 3.1, 4.1
- **Связанные документы:** ТЗ § «Что нужно получить в итоге», текущий `README.md` (структура разделов).
- **Затрагиваемые файлы:** `README.md`.

#### Описание

В существующий `README.md` (после раздела «Команды бота» и перед «Тесты»):

1. Добавить раздел «История диалога»:
   - Где хранится (in-memory, per-user, в `ConversationStore`).
   - Формат сообщений (`role`/`content`).
   - Как ограничивается размер (`HISTORY_MAX_MESSAGES`, FIFO-удаление старейших).
   - Сбрасывается через `/reset` или при рестарте процесса.

2. Добавить раздел «Суммаризация»:
   - Когда срабатывает (`HISTORY_SUMMARY_THRESHOLD`).
   - Как работает: старая часть истории отправляется в LLM с `SUMMARIZATION_PROMPT`, ответ заменяет эту часть как `role: system` сообщение.
   - Что остаётся в истории (последние `kept_tail = 2` сообщения + summary).
   - Если суммаризация падает — диалог не ломается, история обрезается FIFO.

3. В разделе «Команды бота» добавить строку про `/reset`.

4. В разделе «Настройка» (env-переменные) добавить таблицу с новыми переменными `HISTORY_MAX_MESSAGES`, `HISTORY_SUMMARY_THRESHOLD`, `SUMMARIZATION_PROMPT`, `LOG_LLM_CONTEXT` и их default'ами.

5. В разделе «Ограничения MVP» снять строку про «без истории диалога» (если есть) и заменить на актуальную: «История живёт в памяти процесса, теряется при рестарте; персистентного хранилища нет».

#### Definition of Done

- [ ] `README.md` содержит разделы «История диалога» и «Суммаризация» с указанной структурой.
- [ ] В разделе «Команды» упоминается `/reset`.
- [ ] В разделе «Настройка» есть все 4 новых env-переменных с default'ами.
- [ ] `grep -E "история" README.md` показывает упоминание истории как **поддерживаемой** функции (не «без истории»).

---

### Задача 5.2. Обновить `_docs/architecture.md` и `_docs/project-structure.md`

- **Статус:** ToDo
- **Приоритет:** high
- **Объём:** M
- **Зависит от:** Задача 3.1
- **Связанные документы:** существующая структура `_docs/architecture.md` §1–§7, `_docs/project-structure.md` (дерево + таблица «Назначение ключевых модулей»).
- **Затрагиваемые файлы:** `_docs/architecture.md`, `_docs/project-structure.md`.

#### Описание

1. `_docs/architecture.md`:
   - §1 «Общая схема»: добавить узел `ConversationStore` рядом с `aiogram Bot` (через который проходит история перед LLM).
   - §2 «Принципы»: переформулировать «Stateless: каждое сообщение обрабатывается независимо» → «In-memory state: per-user история живёт только в памяти процесса; персистентного хранилища нет».
   - §3.5: переименовать в «Per-user runtime-состояние» и добавить `ConversationStore` рядом с `UserSettingsRegistry`.
   - §3.4: описать `OllamaClient.chat(messages, model)` и `estimate_tokens`; пометить `generate(...)` как «совместимый путь, используется для одношаговых запросов».
   - Добавить §3.8 «Суммаризация диалога» — описание `Summarizer` и порога.
   - §4 «Поток обработки текстового сообщения»: переписать с учётом истории, логирования контекста и условной суммаризации.
   - §7 «Расширяемость»: убрать строку «Добавление истории диалога — ввести отдельный ConversationStore (сейчас явно запрещено ТЗ)»; на её место — «Персистентность истории — добавить адаптер `ConversationStore` поверх БД/Redis».

2. `_docs/project-structure.md`:
   - В дерево `app/services/` добавить `conversation.py` и `summarizer.py`.
   - В дерево `tests/services/` добавить `test_conversation_store.py` и `test_summarizer.py`.
   - В таблицу «Назначение ключевых модулей» добавить строки для двух новых модулей.

#### Definition of Done

- [ ] `_docs/architecture.md` обновлён по 6 пунктам выше; `grep -i "stateless" _docs/architecture.md` либо пусто, либо содержит уточнение про runtime-состояние, не противоречащее истории.
- [ ] `_docs/architecture.md` упоминает `ConversationStore`, `Summarizer`, `chat(messages, …)`.
- [ ] `_docs/project-structure.md` дерево и таблица актуальны.
- [ ] `grep -E "запрещено ТЗ" _docs/architecture.md` пусто.

---

### Задача 5.3. Обновить `_docs/requirements.md` (FR-3, CON-1, новые FR на историю)

- **Статус:** ToDo
- **Приоритет:** high
- **Объём:** M
- **Зависит от:** Задача 3.1, 4.1
- **Связанные документы:** существующий `_docs/requirements.md` §1–§6.
- **Затрагиваемые файлы:** `_docs/requirements.md`.

#### Описание

1. §1 «Функциональные требования»:
   - Переформулировать FR-3: «Каждый пользователь имеет независимую in-memory историю диалога. История ограничена по размеру и при превышении порога суммаризируется. Персистентного хранения нет.»
   - Добавить FR-13 «Перед каждым LLM-запросом полный контекст и его размер логируются (управляется флагом `LOG_LLM_CONTEXT`).»
   - Добавить FR-14 «История ограничена по количеству сообщений (`HISTORY_MAX_MESSAGES`); при превышении самые старые сообщения удаляются.»
   - Добавить FR-15 «При достижении `HISTORY_SUMMARY_THRESHOLD` сообщений старая часть истории сжимается LLM в краткое резюме и заменяет соответствующие сообщения.»
   - Добавить FR-16 «Команда `/reset` стирает историю пользователя и сбрасывает per-user модель и системный промпт.»

2. §3 «Ограничения»:
   - Переформулировать CON-1: «Запрещено персистентное хранение истории диалога (БД, файлы). In-memory история, теряемая при рестарте процесса, разрешена.»

3. §4 «Предположения»:
   - Уточнить ASM-4 (per-user runtime-состояние) — теперь явно включает историю и суммаризированный контекст.

4. §5 «Трассировка»:
   - Обновить строку FR-3 (`app/services/conversation.py`); добавить FR-13/14/15/16 → соответствующие модули и handler'ы.

5. §6 «Критерии верификации»:
   - Добавить строки для FR-13/14/15/16 (T + M).

#### Definition of Done

- [ ] FR-3 переформулирован, добавлены FR-13, FR-14, FR-15, FR-16.
- [ ] CON-1 уточнён (in-memory разрешена).
- [ ] Таблицы трассировки и верификации актуальны.
- [ ] `grep -E "история диалога не хранится" _docs/requirements.md` пусто.

---

### Задача 5.4. Обновить `_docs/commands.md` и `_docs/current-state.md`

- **Статус:** ToDo
- **Приоритет:** medium
- **Объём:** S
- **Зависит от:** Задача 4.1, 5.2
- **Связанные документы:** существующие `_docs/commands.md` (раздел «/reset» помечен как `⏳`), `_docs/current-state.md` §1–§3.
- **Затрагиваемые файлы:** `_docs/commands.md`, `_docs/current-state.md`.

#### Описание

1. `_docs/commands.md`:
   - В сводной таблице сменить статус `/reset` с `⏳` на `✅`.
   - Уточнить описание `/reset`: «стирает историю диалога + сбрасывает per-user модель и системный промпт».
   - Раздел «Произвольный текст» — переписать алгоритм: добавить шаги «append user message», «build messages = [system] + history», «log context», «llm.chat(messages)», «append assistant», «summarize if threshold».

2. `_docs/current-state.md`:
   - §1 «Что работает»: добавить пункты «История диалога per-user (in-memory)», «Лимит истории + FIFO-удаление», «Суммаризация при превышении порога», «`/reset` — очистка истории и сброс настроек», «Логирование контекста перед LLM-запросом».
   - §2.1 (нет `/reset`) — перенести в §6 «История закрытий» с датой и SHA коммита (после закрытия задачи 4.1).
   - §3 «Архитектурные нюансы»: переформулировать «Stateless» (см. Задача 5.2 §2 — то же изменение, но в другом документе). Добавить пункт про оценку токенов как `chars/4` (грубо).

#### Definition of Done

- [ ] `_docs/commands.md` сводная таблица актуальна (`/reset` = ✅), описание команды и алгоритм «Произвольный текст» обновлены.
- [ ] `_docs/current-state.md` §1 содержит ≥ 4 новых пункта про историю/суммаризацию/reset/логирование контекста.
- [ ] `_docs/current-state.md` §2.1 закрыт и перенесён в §6.
- [ ] `_docs/current-state.md` §3 не содержит формулировки «бот не хранит историю диалога» в смысле жёсткого запрета.

---

### Задача 5.5. Финальная приёмка спринта

- **Статус:** ToDo
- **Приоритет:** high
- **Объём:** S
- **Зависит от:** Задачи 5.1, 5.2, 5.3, 5.4
- **Связанные документы:** AC спринта (этот файл, §3), `_board/process.md` § «Чек-лист перед переходом к следующей задаче».
- **Затрагиваемые файлы:** `_board/sprints/03-conversation-context.md` (этот файл — История изменений), `_board/plan.md` (перевод спринта в Closed).

#### Описание

1. Прогнать финальный чек-лист:
   - `pytest -q` зелёный (все добавленные тесты + старые).
   - `python -c "import app"` ОК.
   - `python -c "from app.config import Settings; print(Settings().history_max_messages)"` (с заполненным `.env`) выводит число.
   - Ручная проверка в Telegram (короткий диалог + длинный диалог + `/reset`):
     - Бот помнит первое сообщение в коротком диалоге.
     - В длинном диалоге (≥ `HISTORY_SUMMARY_THRESHOLD` сообщений) лог-файл содержит запись «summarized history».
     - После `/reset` бот забывает контекст.
     - В лог-файле перед каждым LLM-запросом есть строка `llm_context …`.
   - `git status` чист, никаких артефактов (`logs/`, `.env`).
   - `git log -p | grep TELEGRAM_BOT_TOKEN=` возвращает только примеры из `.env.example`.

2. Все AC спринта (§3) проставить как `[x]`.

3. Закрыть спринт по `_board/process.md` §9: статус задач → `Done`, статус спринта → `Closed`, перевод в `_board/plan.md` § «Индекс спринтов» → «Закрытые», обновить сводную таблицу.

#### Definition of Done

- [ ] Все AC спринта (§3) — `[x]`.
- [ ] `pytest -q` зелёный, ручная проверка пройдена.
- [ ] Сводная таблица задач спринта (§9) — все `Done`.
- [ ] `_board/plan.md` обновлён: спринт 03 в «Закрытые», сводная таблица состояния актуальна.
- [ ] § «История изменений спринта» содержит финальную запись с датой закрытия.

---

## 9. Риски и смягчение

| # | Риск                                                                                                  | Смягчение                                                                                                                                  |
|---|-------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------|
| 1 | Локальные модели (`qwen3.5:0.8b`, `deepseek-r1:1.5b`) плохо справляются с суммаризацией на русском     | `SUMMARIZATION_PROMPT` настраиваемый; default-промпт явно требует «только текст резюме»; качество — не критерий приёмки спринта (см. AC). |
| 2 | Большая история → большой prompt → таймаут при суммаризации/основном вызове                            | `HISTORY_MAX_MESSAGES` = 20 (default), порог суммаризации `= 10`; суммаризационная ошибка не валит handler — пользователь получит ответ.   |
| 3 | Логирование полного контекста раскрывает приватный текст пользователя в лог-файле                      | Флаг `LOG_LLM_CONTEXT`; по умолчанию `True` (требование ТЗ §5), но в разделе «История диалога» README отметить privacy-trade-off.          |
| 4 | Переход с `generate(...)` на `chat(...)` в handler рискует сломать старые тесты                       | `generate(...)` оставляем работоспособным; обновляем только `tests/handlers/test_messages.py` и сценарии, пользующиеся новым API.          |
| 5 | Изменение FR-3 / CON-1 в `_docs/requirements.md` — серьёзная правка контракта                         | Спринт открыт по явному ТЗ пользователя (§ «Источник»); правки в Задаче 5.3 формулируются как смягчение, а не отмена контракта.            |
| 6 | Грубая оценка токенов `chars/4` некорректна для кириллицы                                              | Документирована в `_docs/architecture.md` §3.4 / `_docs/current-state.md` §3 как «приблизительная»; точный токенайзер — кандидат roadmap. |

## 10. Сводная таблица задач спринта

| #   | Задача                                                                | Приоритет | Объём | Статус | Зависит от                  |
|-----|-----------------------------------------------------------------------|:---------:|:-----:|:------:|:---------------------------:|
| 1.1 | Расширить `Settings` и `.env.example` параметрами истории             | high      | S     | Done   | —                           |
| 1.2 | `ConversationStore` — in-memory история per-user                      | high      | M     | Progress | Задача 1.1                |
| 2.1 | `OllamaClient.chat(messages, model)` поверх Ollama chat-API           | high      | M     | ToDo   | Задача 1.2                  |
| 2.2 | `Summarizer` — сжатие старой части диалога через LLM                  | high      | S     | ToDo   | Задача 2.1                  |
| 3.1 | Handler текста: контекст, логирование, `chat`, обновление истории     | high      | L     | ToDo   | Задачи 1.2, 2.1, 2.2        |
| 4.1 | Команда `/reset` и обновление справочных текстов                      | medium    | S     | ToDo   | Задача 3.1                  |
| 5.1 | Обновить `README.md`: «История диалога», «Суммаризация»               | high      | S     | ToDo   | Задачи 3.1, 4.1             |
| 5.2 | Обновить `_docs/architecture.md` и `_docs/project-structure.md`       | high      | M     | ToDo   | Задача 3.1                  |
| 5.3 | Обновить `_docs/requirements.md` (FR-3, CON-1, новые FR)              | high      | M     | ToDo   | Задачи 3.1, 4.1             |
| 5.4 | Обновить `_docs/commands.md` и `_docs/current-state.md`               | medium    | S     | ToDo   | Задачи 4.1, 5.2             |
| 5.5 | Финальная приёмка спринта                                             | high      | S     | ToDo   | Задачи 5.1, 5.2, 5.3, 5.4   |

## 11. История изменений спринта

- **2026-04-26** — спринт открыт, ветка `feature/conversation-context` создана от `main`.
- **2026-04-26** — усилены правила тестирования (обязательные unit-тесты для кода в `app/`, зелёный `pytest -q` перед коммитом) — `docs(rules): require unit tests for new code in app/ and green pytest before commit`.
- **2026-04-26** — закрыта задача 1.1: расширен `Settings` полями `history_max_messages`, `history_summary_threshold`, `summarization_prompt`, `log_llm_context` и валидатором лимитов; `.env.example` дополнен секцией «Conversation context» (коммит `feat(config): add conversation history settings (max, summary threshold, prompt, log flag)`).
