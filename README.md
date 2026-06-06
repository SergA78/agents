# Агент конкурентной разведки (Competitor Intelligence Agent)

ИИ-агент, который **непрерывно собирает данные о конкурентах** (новости, пресс-релизы, отзывы, изменения цен и вакансии) для компаний **Apple** и **Microsoft**, складывает их в **RAG-хранилище** (PostgreSQL + pgvector) и по запросу формирует **аналитические сводки на русском языке** и **алерты об изменениях**.

Оркестрация построена на **LangGraph**: периодический сбор (scheduled), дедупликация, генерация отчётов с трендами и алертами, а также отдельный граф для аналитического Q&A. На старте выполняется **SEED** — первичный сбор данных за последние 30 дней.

> Это MVP. Цель — продемонстрировать рабочий end-to-end контур конкурентной разведки на открытых веб-источниках.

---

## Оглавление

1. [Возможности](#возможности)
2. [Архитектура](#архитектура)
   - [Компоненты системы](#компоненты-системы)
   - [Графы LangGraph](#графы-langgraph)
   - [Поток запроса `/query`](#поток-запроса-query)
3. [Структура проекта](#структура-проекта)
4. [Установка и запуск](#установка-и-запуск)
   - [Предварительные требования](#предварительные-требования)
   - [Запуск через Docker Compose](#запуск-через-docker-compose)
   - [Что происходит на старте](#что-происходит-на-старте)
   - [Проверка работоспособности](#проверка-работоспособности)
   - [Веб-интерфейс чата](#веб-интерфейс-чата)
   - [Примеры запросов (curl)](#примеры-запросов-curl)
   - [Выбор LLM-провайдера (OpenAI / Ollama)](#выбор-llm-провайдера-openai--ollama)
   - [Использование Ollama](#использование-ollama)
   - [Настройка Langfuse](#настройка-langfuse)
   - [Локальный запуск без Docker (опционально)](#локальный-запуск-без-docker-опционально)
5. [Конфигурация](#конфигурация)
6. [Описание API](#описание-api)
7. [Тестирование](#тестирование)
   - [Философия тестов (offline-моки + self-skip)](#философия-тестов-offline-моки--self-skip)
   - [Состав тестов](#состав-тестов)
   - [Маркеры pytest](#маркеры-pytest)
   - [Команды запуска](#команды-запуска)
8. [Кому подойдёт этот агент](#кому-подойдёт-этот-агент)
9. [Edge cases и обработка ошибок](#edge-cases-и-обработка-ошибок)
10. [Почему недостаточно обычного детерминированного пайплайна](#почему-недостаточно-обычного-детерминированного-пайплайна)
11. [Критерии оценки результативности](#критерии-оценки-результативности)
12. [Источники данных и интеграции](#источники-данных-и-интеграции)
13. [Ограничения MVP и возможные улучшения](#ограничения-mvp-и-возможные-улучшения)
14. [Лицензия и дисклеймер](#лицензия-и-дисклеймер)

---

## Возможности

- **Непрерывный автоматический сбор** данных о конкурентах по расписанию (по умолчанию каждые 6 часов) через планировщик APScheduler.
- **Первичный сбор (SEED)** за последние 30 дней при старте сервиса — база сразу наполняется данными.
- **Пять категорий данных** по каждой компании: новости (`news`), пресс-релизы (`press_release`), отзывы (`review`), изменения цен (`price`), вакансии (`job`).
- **Веб-поиск через SearXNG** (метапоиск) с обёрткой в виде MCP-совместимого инструмента поиска.
- **RAG-хранилище** на PostgreSQL + pgvector: документы хранятся вместе с векторными эмбеддингами для семантического поиска.
- **Двухуровневая дедупликация**: точная по `content_hash` (в батче и в БД) + утилита оценки близких дубликатов по косинусной близости.
- **Аналитические отчёты на русском** с автоматическим выделением трендов (генерируются LLM, сохраняются в БД).
- **Алерты об изменениях** на основе эвристик: изменение цены (high), активность по найму (medium), всплеск новостной активности (medium/high).
- **Аналитический Q&A** (`/query`): семантический поиск по хранилищу + ответ LLM на русском со ссылками на источники.
- **Встроенный веб-чат** (vanilla HTML/CSS/JS, без CDN): аналитические сводки можно запрашивать прямо из браузера на `http://localhost:8000/`, не прибегая к `curl`.
- **Трассировка** пользовательских запросов в Langfuse (устойчиво к отсутствию/недоступности Langfuse — работает в режиме no-op).
- **Высокая отказоустойчивость**: все узлы графа и внешние вызовы защищены — ошибки накапливаются в `errors`, но не «роняют» пайплайн; есть ретраи (tenacity) и фолбэки.
- **Выбор LLM-провайдера: OpenAI или Ollama** — провайдер выбирается **независимо** для чата (`CHAT_PROVIDER`) и для эмбеддингов (`EMBEDDING_PROVIDER`). Ollama предоставляет OpenAI-совместимый API, поэтому используется тот же `openai` SDK.
- **Любой OpenAI-совместимый LLM**: провайдер, модель и эмбеддинги задаются через `.env`.

---

## Архитектура

Система состоит из приложения FastAPI с оркестрацией LangGraph и набора инфраструктурных сервисов, поднимаемых через Docker Compose. Приложение отдаёт встроенный веб-чат (браузер → `app` → `/query`) и взаимодействует с PostgreSQL/pgvector (хранилище RAG), SearXNG (веб-поиск), LLM-провайдером (чат + эмбеддинги, **OpenAI или Ollama**) и Langfuse (трассировка).

### Компоненты системы

```mermaid
flowchart TB
    user([Пользователь — браузер / API-клиент])

    subgraph compose["Docker Compose: сеть ci_net"]
        app["app — FastAPI / uvicorn\nвеб-чат (/) + REST\nLangGraph + APScheduler\nпорт 8000"]
        pg[("postgres\npgvector/pgvector:pg16\nпорт 5432\nтаблицы: documents, reports,\nalerts, collection_runs")]
        searx["searxng\nметапоиск, JSON API\nпорт 8080"]
        lfweb["langfuse-web\nlangfuse/langfuse:2\nпорт 3000"]
        lfdb[("langfuse-db\npostgres:16")]
        ollama["ollama (опционально)\nollama/ollama:latest\nпорт 11434\nпрофиль compose: ollama"]
    end

    llm["LLM-провайдер\nOpenAI-совместимый API\nchat + embeddings\nOpenAI (внешний) ИЛИ Ollama"]

    user -->|веб-чат / HTTP REST| app
    app -->|SQLAlchemy + pgvector| pg
    app -->|HTTP JSON поиск| searx
    app -->|chat / embeddings| llm
    llm -.->|вариант: OpenAI| openai["OpenAI API (внешний)"]
    llm -.->|вариант: Ollama| ollama
    app -->|трассировка /query| lfweb
    lfweb --> lfdb
```

**Краткое описание компонентов:**

- **app** — ядро системы: встроенный веб-чат (`GET /`), REST API на FastAPI, графы LangGraph (сбор и Q&A), планировщик APScheduler, слой RAG, генератор отчётов и алертов.
- **postgres (pgvector)** — основное хранилище: документы с эмбеддингами, отчёты, алерты и журнал запусков сбора.
- **searxng** — метапоисковый движок, отдаёт результаты в JSON; единая точка доступа к открытому вебу.
- **langfuse-web + langfuse-db** — приём и хранение трасс пользовательских запросов `/query`.
- **LLM-провайдер** — OpenAI-совместимый API для генерации текстов (chat) и построения эмбеддингов. Провайдером может быть **OpenAI** (внешний) или **Ollama** (локальный, опциональный сервис `ollama`), причём чат и эмбеддинги выбираются независимо.
- **ollama (опционально)** — локальный LLM-сервис с OpenAI-совместимым API; включается профилем compose `ollama` и обычным `docker compose up` не запускается.
- **tests (опционально)** — одноразовый сервис для прогона `pytest` против поднятого стека (`APP_TEST_BASE_URL=http://app:8000`); включается профилем compose `test` и обычным `docker compose up` не запускается. Подробнее — в разделе [Тестирование](#тестирование).

### Графы LangGraph

В системе **два независимых графа**, оба линейные.

**Граф сбора** (`collect → dedup → store → report → alert`) — запускается планировщиком, сидом и эндпоинтом `/collect`:

```mermaid
flowchart LR
    START((START)) --> collect["collect_node\nвеб-поиск по\n(company × category)"]
    collect --> dedup["dedup_node\nудаление пустых и\nдубликатов по content_hash"]
    dedup --> store["store_node\nэмбеддинги + запись\nв pgvector (дедуп по БД)"]
    store --> report["report_node\nгенерация отчёта LLM\n+ тренды, сохранение"]
    report --> alert["alert_node\nэвристические алерты,\nсохранение"]
    alert --> END((END))
```

**Граф запроса** (`retrieve → answer`) — запускается эндпоинтом `/query`:

```mermaid
flowchart LR
    QSTART((START)) --> retrieve["retrieve_node\nsimilarity_search\nпо pgvector"]
    retrieve --> answer["answer_node\nответ LLM на русском\nсо ссылками на источники"]
    answer --> QEND((END))
```

Все узлы устойчивы к ошибкам: исключения перехватываются, записываются в `state['errors']`, и узел корректно возвращает частичное состояние — единичный сбой не прерывает граф.

### Поток запроса `/query`

```mermaid
sequenceDiagram
    actor U as Пользователь
    participant WEB as Веб-чат (браузер)
    participant API as FastAPI (/query)
    participant RQ as run_query (@traced)
    participant G as Query Graph
    participant RET as retrieve_node
    participant STORE as RAGStore
    participant PG as PostgreSQL + pgvector
    participant LLM as LLM-провайдер
    participant LF as Langfuse

    U->>WEB: вводит вопрос (компания?, период?)
    WEB->>API: POST /query {query, company?, period_days?}
    API->>API: валидация (query не пустой)
    API->>RQ: run_query(...)
    RQ->>LF: старт трассы
    RQ->>G: invoke(state)
    G->>RET: retrieve_node
    RET->>STORE: similarity_search(query, company, since)
    STORE->>LLM: embed_one(query)
    alt эмбеддинг получен
        STORE->>PG: ORDER BY cosine_distance LIMIT k
    else ошибка эмбеддинга
        STORE->>PG: фолбэк: recent_documents (свежие)
    end
    PG-->>STORE: документы
    STORE-->>RET: список документов
    RET-->>G: retrieved[]
    G->>LLM: answer_node — chat(system, user+context)
    LLM-->>G: аналитический ответ (RU)
    G-->>RQ: {answer, retrieved}
    RQ->>LF: завершение трассы + flush
    RQ-->>API: {answer, retrieved}
    API-->>WEB: {answer, sources:[urls]}
    WEB-->>U: ответ (текст) + кликабельные ссылки
```

Если релевантных документов не найдено, `answer_node` возвращает честный ответ «недостаточно данных», не обращаясь к LLM с пустым контекстом.

---

## Структура проекта

```text
E:\Git\agents
├── docker-compose.yml          # Сервисы: postgres, searxng, langfuse-db,
│                               # langfuse-web, app; опциональный ollama (профиль ollama);
│                               # опциональный tests (профиль test) — прогон pytest
│                               # против поднятого стека (APP_TEST_BASE_URL=http://app:8000);
│                               # сеть ci_net; тома pgdata, langfuse_pgdata, ollama_data
├── Dockerfile                  # python:3.11-slim; установка requirements;
│                               # запуск uvicorn app.main:app
├── requirements.txt            # Зависимости (langgraph, langchain*, openai, fastapi,
│                               # uvicorn, pydantic*, sqlalchemy>=2, psycopg[binary],
│                               # pgvector, apscheduler, langfuse, httpx, bs4, lxml,
│                               # tenacity, numpy);
│                               # dev/тестовые зависимости: pytest, pytest-timeout
│                               # (httpx уже присутствует — используется в TestClient)
├── pytest.ini                  # Конфигурация pytest: маркеры integration/llm/e2e
├── tests/                      # Автотест-набор на pytest (см. раздел «Тестирование»)
│   ├── __init__.py
│   ├── conftest.py             # Общие фикстуры; autouse self-skip ресурсо-зависимых
│   │                           # тестов; базовый URL из APP_TEST_BASE_URL/BASE_URL
│   ├── test_installation.py    # «приложение установлено»: импорт зависимостей и
│   │                           # пакета app, наличие файлов поставки
│   ├── test_modules.py         # «все модули подключены»: импорт app.*, FastAPI-маршруты,
│   │                           # компиляция обоих графов LangGraph, таблицы БД, CATEGORIES
│   ├── test_app_running.py     # «приложение запущено»: офлайн через TestClient + e2e по HTTP
│   ├── test_llm.py             # «запросы к LLM проходят»: юнит на моках + реальный вызов (llm)
│   ├── test_seed.py            # «для SEED собрана статистика»: run_seed() (моки) + e2e
│   └── test_analytics_queries.py  # «≥3 аналитических запроса» + объективность сводок
├── .env.example                # Шаблон переменных окружения
├── searxng/
│   └── settings.yml            # Конфиг SearXNG (включён JSON-формат)
└── app/
    ├── main.py                 # FastAPI-приложение; startup (init_db, scheduler,
    │                           # seed в фоне) / shutdown; все REST-эндпоинты;
    │                           # монтирует /static (StaticFiles, если папка есть);
    │                           # GET / = веб-чат (index.html), GET /info = JSON-листинг
    ├── static/                 # Встроенный веб-чат (vanilla, без CDN)
    │   ├── index.html          # Разметка чата: шапка, выбор компании/периода,
    │   │                       # область сообщений, textarea + кнопка «Отправить»
    │   ├── styles.css          # Стили веб-чата
    │   └── app.js              # Логика: POST /query (same origin), рендер answer
    │                           # (textContent, XSS-safe) + sources (ссылки),
    │                           # индикатор загрузки, ошибки, блокировка кнопки
    ├── config.py               # pydantic-settings Settings + singleton settings;
    │                           # CATEGORIES = [news, press_release, review, price, job]
    ├── llm.py                  # LLM-клиент(ы) OpenAI/Ollama: get_chat_client(),
    │                           # get_embed_client() (кэш per-purpose), get_client()
    │                           # (совместимость); chat(), embed(), embed_one();
    │                           # выбор модели по провайдеру; ретраи через tenacity
    ├── tracing.py              # Langfuse-хелпер: get_langfuse(), декоратор traced(name),
    │                           # flush(); no-op при недоступности Langfuse
    ├── seed.py                 # run_seed(): init_db + run_collection(month) +
    │                           # report/alerts по компаниям; запуск: python -m app.seed
    ├── db/
    │   ├── session.py          # engine, SessionLocal, get_session(), Base
    │   ├── models.py           # ORM-модели: Document, Report, Alert, CollectionRun
    │   └── init_db.py          # init_db(): CREATE EXTENSION vector + create_all;
    │                           # ретраи коннекта к БД
    ├── rag/
    │   ├── dedup.py            # content_hash(), normalize_text(),
    │   │                       # is_near_duplicate() (косинусная близость)
    │   └── store.py            # RAGStore: add_documents() (дедуп по хэшу + батч-эмбеддинги),
    │                           # similarity_search() (cosine_distance + фолбэк на свежие),
    │                           # recent_documents(), count()
    ├── search/
    │   ├── searxng.py          # SearxngClient.search() по JSON API; ретраи;
    │   │                       # при ошибке возвращает [] (не падает)
    │   ├── queries.py          # build_queries(company, categories): шаблоны запросов
    │   │                       # по категориям, time_range="month"
    │   └── mcp_client.py       # SearxngMCPClient + get_search_tool() —
    │                           # абстракция поиска для узлов графа
    ├── graph/
    │   ├── state.py            # TypedDict CollectionState и QueryState
    │   ├── nodes.py            # Узлы: collect/dedup/store/report/alert (сбор),
    │   │                       # retrieve/answer (Q&A); все @traced и устойчивы к ошибкам
    │   ├── builder.py          # build_collection_graph(), build_query_graph();
    │   │                       # run_collection(...), run_query(...) (@traced)
    │   └── scheduler.py        # APScheduler BackgroundScheduler; start_scheduler()
    │                           # (интервал=COLLECT_INTERVAL_HOURS, coalesce,
    │                           # max_instances=1), shutdown_scheduler()
    └── reports/
        └── generator.py        # generate_report() -> {company, period_days, summary,
                                # trends, doc_count} + сохранение Report;
                                # detect_alerts() — эвристики price/hiring/news_spike +
                                # сохранение Alert
```

**Модели данных (`app/db/models.py`):**

| Модель | Таблица | Назначение |
|---|---|---|
| `Document` | `documents` | Собранный документ: `company`, `category`, `title`, `url`, `content`, `content_hash` (uniq), `source`, `published_at`, `collected_at`, `doc_metadata`, `embedding` (`Vector`). |
| `Report` | `reports` | Аналитический отчёт по компании: `company`, `period_days`, `summary`, `trends`, `created_at`. |
| `Alert` | `alerts` | Алерт об изменении: `company`, `category`, `alert_type`, `severity`, `message`, `document_id`, `created_at`. |
| `CollectionRun` | `collection_runs` | Журнал запусков сбора. |

---

## Установка и запуск

### Предварительные требования

- **Docker** и **Docker Compose** (v2, команда `docker compose`).
- **Доступ к OpenAI-совместимому LLM-провайдеру**: ключ API и базовый URL (по умолчанию — официальный OpenAI; можно указать любой совместимый провайдер).
- Свободные порты на хосте: `8000` (app), `5432` (postgres), `8080` (searxng), `3000` (langfuse).

### Запуск через Docker Compose

1. Скопируйте шаблон переменных окружения и заполните его:

   ```bash
   cp .env.example .env
   ```

2. Откройте `.env` и заполните как минимум доступ к LLM:

   ```env
   OPENAI_API_KEY=sk-...                  # ключ вашего LLM-провайдера
   OPENAI_BASE_URL=https://api.openai.com/v1
   OPENAI_MODEL=gpt-4o-mini
   EMBEDDING_MODEL=text-embedding-3-small
   EMBEDDING_DIM=1536
   ```

   При желании настройте список компаний (`COMPANIES`), окно сида (`SEED_DAYS`), интервал сбора (`COLLECT_INTERVAL_HOURS`) и ключи Langfuse (см. раздел [Настройка Langfuse](#настройка-langfuse)).

3. Соберите и поднимите все сервисы:

   ```bash
   docker compose up -d --build
   ```

4. Посмотрите логи приложения, чтобы убедиться, что старт прошёл и сид запустился:

   ```bash
   docker compose logs -f app
   ```

5. Откройте [http://localhost:8000/](http://localhost:8000/) — это встроенный **веб-чат** для аналитических запросов (см. раздел [Веб-интерфейс чата](#веб-интерфейс-чата)).

### Что происходит на старте

При запуске сервиса `app` (событие `startup` в `app/main.py`):

1. **Настройка логирования** по уровню `LOG_LEVEL`.
2. **`init_db()`** — создаётся расширение `vector` в PostgreSQL и все таблицы (`create_all`); коннект к БД выполняется с ретраями.
3. **`start_scheduler()`** — запускается APScheduler `BackgroundScheduler` с периодическим сбором каждые `COLLECT_INTERVAL_HOURS` часов (по умолчанию — 6), с `coalesce=True` и `max_instances=1`.
4. **`run_seed()` в фоновом потоке** — не блокирует старт; выполняет первичный сбор за последние `SEED_DAYS` дней (по умолчанию — 30) с генерацией отчётов и алертов по каждой компании.

При остановке (событие `shutdown`): останавливается планировщик и сбрасывается буфер трассировки Langfuse (`tracing.flush()`).

> Сид может занять некоторое время и зависит от доступности SearXNG и LLM. Сервис при этом уже отвечает на запросы; данные появляются по мере наполнения хранилища.

### Проверка работоспособности

- **Веб-чат:** [http://localhost:8000/](http://localhost:8000/) — встроенный браузерный интерфейс для аналитических запросов.
- **JSON-описание эндпоинтов:** [http://localhost:8000/info](http://localhost:8000/info) — имя проекта и список эндпоинтов.
- **Swagger UI:** [http://localhost:8000/docs](http://localhost:8000/docs) — интерактивная документация API.
- **Health-check:** [http://localhost:8000/health](http://localhost:8000/health) → `{"status":"ok"}`.
- **Langfuse:** [http://localhost:3000](http://localhost:3000) — UI трассировки.
- **SearXNG:** [http://localhost:8080](http://localhost:8080) — поисковый движок.

### Веб-интерфейс чата

Самый простой способ работы с агентом — встроенный **веб-чат** (vanilla HTML/CSS/JS, без внешних CDN), который отдаётся самим FastAPI:

1. Откройте [http://localhost:8000/](http://localhost:8000/) в браузере.
2. (Опционально) выберите компанию в селекторе («Все компании» / Apple / Microsoft) и задайте «Период, дней».
3. Введите вопрос в поле ввода и нажмите **«Отправить»** (`Enter` — отправка, `Shift+Enter` — перенос строки).

Под капотом чат вызывает тот же эндпоинт `POST /query` (тот же origin) с телом `{query, company|null, period_days|null}` и отображает:

- текст ответа (`answer`) как обычный текст (XSS-safe, через `textContent`);
- список источников (`sources`) в виде кликабельных ссылок;
- индикатор загрузки и сообщения об ошибках (кнопка «Отправить» блокируется на время запроса).

> Статические файлы веб-чата лежат в `app/static/` и монтируются на `/static`. Если папка отсутствует, `GET /` возвращает JSON-описание (как `/info`).

### Примеры запросов (curl)

**Ручной запуск сбора** (`/collect`). Тело необязательно — без параметров используются настройки по умолчанию:

```bash
curl -X POST http://localhost:8000/collect \
  -H "Content-Type: application/json" \
  -d '{"companies": ["Apple"], "categories": ["news", "price"], "time_range": "week"}'
```

Пример ответа:

```json
{
  "store_stats": {"received": 24, "inserted": 18, "duplicates": 6},
  "raw_count": 30,
  "deduped_count": 24,
  "alerts": [
    {"company": "Apple", "category": "price", "alert_type": "price_change", "severity": "high", "message": "Обнаружено изменение/упоминание цены: ..."}
  ],
  "errors": []
}
```

**Аналитический Q&A** (`/query`):

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Какие ключевые изменения у Microsoft за последний месяц?", "company": "Microsoft", "period_days": 30}'
```

Пример ответа:

```json
{
  "answer": "За последний месяц Microsoft... (аналитический текст на русском со ссылками)",
  "sources": [
    "https://example.com/news/1",
    "https://example.com/press/2"
  ]
}
```

**Последние отчёты** (`/reports`):

```bash
curl "http://localhost:8000/reports?company=Apple"
```

**Последние алерты** (`/alerts`):

```bash
curl "http://localhost:8000/alerts?company=Microsoft"
```

**Статистика по документам** (`/documents/stats`):

```bash
curl "http://localhost:8000/documents/stats"
```

Пример ответа:

```json
{"total": 142, "per_company": {"Apple": 78, "Microsoft": 64}}
```

### Выбор LLM-провайдера (OpenAI / Ollama)

LLM-доступ поддерживает **двух провайдеров — OpenAI и Ollama**, выбираемых **независимо** для чата/анализа и для эмбеддингов. Ollama предоставляет OpenAI-совместимый API, поэтому в обоих случаях используется один и тот же `openai` SDK — различаются лишь `base_url`, `api_key` и имя модели. В `app/llm.py` для этого добавлены кэшируемые `get_chat_client()` и `get_embed_client()` (per-purpose); прежний `get_client()` сохранён (возвращает chat-клиент) для обратной совместимости, а публичные `chat()`, `embed()`, `embed_one()` не меняли сигнатур — имя модели подбирается автоматически по провайдеру.

Провайдеры задаются в `.env` через `CHAT_PROVIDER` и `EMBEDDING_PROVIDER` (`openai` | `ollama`). Любой облачный/локальный **OpenAI-совместимый** провайдер можно подключить, изменив переменные `OPENAI_*`:

```env
OPENAI_BASE_URL=https://your-provider.example/v1   # базовый URL совместимого API
OPENAI_API_KEY=your-key                            # ключ провайдера
OPENAI_MODEL=your-chat-model                        # модель для chat
EMBEDDING_MODEL=your-embedding-model                # модель эмбеддингов
EMBEDDING_DIM=1536                                  # размерность эмбеддингов модели
```

> Важно: значение `EMBEDDING_DIM` должно совпадать с размерностью выбранной модели эмбеддингов, так как оно определяет тип колонки `embedding` в pgvector (`text-embedding-3-small` = 1536, `nomic-embed-text` = 768). При смене **провайдера/модели эмбеддингов** выставьте корректный `EMBEDDING_DIM` и пересоздайте БД (удалите том `pgdata`) — размерность вектора фиксируется при создании схемы.

Сценарии «совместно» допустимы: например, Ollama для чата (`CHAT_PROVIDER=ollama`) и OpenAI для эмбеддингов (`EMBEDDING_PROVIDER=openai`), или наоборот.

После правок перезапустите приложение:

```bash
docker compose up -d --build app
```

### Использование Ollama

Ollama — это **опциональный** сервис Docker Compose (образ `ollama/ollama:latest`, порт `11434`, том `ollama_data`), включаемый профилем compose `ollama`. Обычный `docker compose up` его **не** запускает, и сервис `app` не имеет жёсткого `depends_on` на `ollama` (он gated профилем).

Запуск с Ollama и загрузка моделей:

```bash
docker compose --profile ollama up -d
docker compose exec ollama ollama pull llama3.1
docker compose exec ollama ollama pull nomic-embed-text
```

Затем в `.env` выберите Ollama для нужной задачи (для чата и/или эмбеддингов независимо):

```env
CHAT_PROVIDER=ollama
EMBEDDING_PROVIDER=ollama
OLLAMA_BASE_URL=http://ollama:11434/v1
OLLAMA_API_KEY=ollama
OLLAMA_MODEL=llama3.1
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
EMBEDDING_DIM=768
```

> При переходе на `nomic-embed-text` обязательно выставьте `EMBEDDING_DIM=768` и пересоздайте БД (удалите том `pgdata`), иначе размерность вектора не совпадёт со схемой.

В compose-файле есть закомментированный блок NVIDIA GPU (опционально, требует `nvidia-container-toolkit`); без GPU Ollama работает на CPU.

### Настройка Langfuse

Трассировка пользовательских запросов `/query` опциональна и включается заполнением ключей:

1. Откройте [http://localhost:3000](http://localhost:3000) и создайте аккаунт/проект в Langfuse.
2. В настройках проекта получите **Public Key** и **Secret Key**.
3. Впишите их в `.env`:

   ```env
   LANGFUSE_PUBLIC_KEY=pk-lf-...
   LANGFUSE_SECRET_KEY=sk-lf-...
   LANGFUSE_HOST=http://langfuse-web:3000
   ```

4. Перезапустите приложение:

   ```bash
   docker compose up -d app
   ```

Если ключи не заданы или Langfuse недоступен, трассировка работает в режиме **no-op** — это не влияет на функциональность.

### Локальный запуск без Docker (опционально)

1. Создайте виртуальное окружение и установите зависимости:

   ```bash
   pip install -r requirements.txt
   ```

2. Поднимите **PostgreSQL с pgvector** и **SearXNG** (например, локально или в отдельных контейнерах) и укажите их адреса в `.env`:

   ```env
   DATABASE_URL=postgresql+psycopg://ci_agent:ci_agent_pass@localhost:5432/ci_agent
   SEARXNG_URL=http://localhost:8080
   ```

3. (Опционально) выполните первичный сбор вручную:

   ```bash
   python -m app.seed
   ```

4. Запустите приложение:

   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```

---

## Конфигурация

Все переменные читаются из окружения и/или файла `.env` (см. `app/config.py`). Неизвестные ключи игнорируются; имена регистронезависимы.

| Переменная | Описание | Значение по умолчанию |
|---|---|---|
| `CHAT_PROVIDER` | Провайдер для чата/анализа: `openai` \| `ollama`. | `openai` |
| `EMBEDDING_PROVIDER` | Провайдер для эмбеддингов: `openai` \| `ollama` (выбирается независимо от чата). | `openai` |
| `OPENAI_API_KEY` | Ключ доступа к OpenAI-совместимому LLM-провайдеру. | `""` (пусто) |
| `OPENAI_BASE_URL` | Базовый URL совместимого API. | `https://api.openai.com/v1` |
| `OPENAI_MODEL` | Модель для генерации текста (chat) при `CHAT_PROVIDER=openai`. | `gpt-4o-mini` |
| `EMBEDDING_MODEL` | Модель эмбеддингов при `EMBEDDING_PROVIDER=openai`. | `text-embedding-3-small` |
| `OLLAMA_BASE_URL` | OpenAI-совместимый endpoint Ollama. | `http://ollama:11434/v1` |
| `OLLAMA_API_KEY` | Ollama игнорирует ключ, но OpenAI SDK требует непустое значение. | `ollama` |
| `OLLAMA_MODEL` | Модель чата при `CHAT_PROVIDER=ollama`. | `llama3.1` |
| `OLLAMA_EMBEDDING_MODEL` | Модель эмбеддингов при `EMBEDDING_PROVIDER=ollama`. | `nomic-embed-text` |
| `EMBEDDING_DIM` | Размерность эмбеддингов (тип колонки `embedding`); должна соответствовать модели эмбеддингов: `text-embedding-3-small` = 1536, `nomic-embed-text` = 768. При смене провайдера/модели эмбеддингов пересоздайте БД (том `pgdata`). | `1536` |
| `DATABASE_URL` | Строка подключения к PostgreSQL (SQLAlchemy + psycopg). | `postgresql+psycopg://ci_agent:ci_agent_pass@postgres:5432/ci_agent` |
| `POSTGRES_*` | Параметры инициализации PostgreSQL (user/password/db) для контейнера. | см. `.env.example` / `docker-compose.yml` |
| `SEARXNG_URL` | Базовый URL SearXNG. | `http://searxng:8080` |
| `LANGFUSE_PUBLIC_KEY` | Публичный ключ Langfuse (опционально). | `""` (пусто) |
| `LANGFUSE_SECRET_KEY` | Секретный ключ Langfuse (опционально). | `""` (пусто) |
| `LANGFUSE_HOST` | URL Langfuse. | `http://langfuse-web:3000` |
| `COMPANIES` | Список компаний-конкурентов через запятую. | `Apple,Microsoft` |
| `SEED_DAYS` | Окно первичного сбора (дней) и период отчёта. | `30` |
| `COLLECT_INTERVAL_HOURS` | Интервал периодического сбора (часов). | `6` |
| `APP_PORT` | Порт приложения. | `8000` |
| `LOG_LEVEL` | Уровень логирования. | `INFO` |

Категории данных фиксированы в коде (`app/config.py`): `CATEGORIES = ["news", "press_release", "review", "price", "job"]`.

---

## Описание API

| Метод | Путь | Назначение |
|---|---|---|
| `GET` | `/` | Встроенный веб-чат (HTML, `index.html`); при отсутствии файла — фолбэк на JSON-листинг (как `/info`). |
| `GET` | `/info` | Имя проекта и список доступных эндпоинтов (JSON). |
| `GET` | `/static/*` | Статические ассеты веб-чата (если папка `app/static` присутствует). |
| `GET` | `/health` | Проверка работоспособности (`{"status":"ok"}`). |
| `POST` | `/collect` | Синхронный запуск сбора; возвращает статистику хранилища, счётчики и алерты. |
| `POST` | `/query` | Аналитический Q&A по собранным данным; ответ на русском со ссылками. Трассируется в Langfuse. |
| `GET` | `/reports?company=` | Последние отчёты (опционально по компании). |
| `GET` | `/alerts?company=` | Последние алерты (опционально по компании). |
| `GET` | `/documents/stats` | Счётчики документов по компаниям и общий итог. |

### `POST /collect`

**Тело запроса** (все поля опциональны):

```json
{
  "companies": ["Apple", "Microsoft"],
  "categories": ["news", "press_release", "review", "price", "job"],
  "time_range": "month"
}
```

**Ответ:** `store_stats` (`received`/`inserted`/`duplicates`), `raw_count`, `deduped_count`, `alerts`, `errors`.

### `POST /query`

**Тело запроса:**

```json
{
  "query": "Что нового у Apple с ценами?",
  "company": "Apple",
  "period_days": 30
}
```

- `query` — обязателен и не должен быть пустым (иначе `400`).
- `company`, `period_days` — опциональны (сужают и ограничивают по времени выборку документов).

**Ответ:** `{"answer": "...", "sources": ["url", ...]}`.

### `GET /reports` / `GET /alerts` / `GET /documents/stats`

- `/reports` — массив отчётов: `id`, `company`, `period_days`, `summary`, `trends`, `created_at`.
- `/alerts` — массив алертов: `id`, `company`, `category`, `alert_type`, `severity`, `message`, `document_id`, `created_at`.
- `/documents/stats` — `{"total": N, "per_company": {company: count}}`.

### Веб-чат (`GET /`)

Основной способ взаимодействия пользователя — встроенный веб-чат, который отдаётся на корне `/` (HTML из `app/static/index.html`, ассеты — на `/static`). Он не добавляет новых серверных эндпоинтов: из браузера он вызывает тот же `POST /query` (same origin) и отображает `answer` (как текст, XSS-safe) и `sources` (кликабельные ссылки). JSON-листинг эндпоинтов, ранее доступный на `/`, теперь находится на `/info` (содержимое не изменилось). Swagger по-прежнему на `/docs`. Подробнее — в разделе [Веб-интерфейс чата](#веб-интерфейс-чата).

---

## Тестирование

Проект снабжён автотест-набором на **pytest** (каталог `tests/`, конфигурация `pytest.ini`). Тестовые зависимости (`pytest`, `pytest-timeout`) добавлены в `requirements.txt`; `httpx` присутствовал и ранее — он используется в `TestClient` и для HTTP-проверок. В `docker-compose.yml` добавлен **опциональный** сервис `tests` под профилем compose `test` для прогона набора против поднятого стека.

### Философия тестов (offline-моки + self-skip)

Обычный запуск `pytest` работает **где угодно**:

- **Юнит-тесты выполняются полностью офлайн** — внешние ресурсы (БД, SearXNG, LLM, HTTP-сервер приложения) замоканы, поэтому такие тесты не требуют поднятого стека.
- **Тесты, которым нужны внешние ресурсы, сами пропускаются (skip)**, если ресурс недоступен, — они **никогда не падают** из-за отсутствия инфраструктуры. Авто-skip реализован `autouse`-фикстурой (`tests/conftest.py`).
- **Базовый URL приложения** берётся из переменной окружения `APP_TEST_BASE_URL` (или `BASE_URL`; по умолчанию `http://localhost:8000`). В docker-сервисе `tests` он равен `http://app:8000`.

### Состав тестов

| Файл | Что проверяет (вопрос пользователя) |
|---|---|
| `tests/test_installation.py` | «приложение установлено» — все сторонние зависимости импортируются, пакет `app` и подмодули импортируются, присутствуют файлы поставки (`docker-compose.yml`, `Dockerfile`, `requirements.txt`, `.env.example`, `README.md`, `pytest.ini`, `app/static/*`). |
| `tests/test_modules.py` | «все необходимые модули подключены» — импорт всех модулей `app.*`; FastAPI-маршруты (`/`, `/info`, `/health`, `/collect`, `/query`, `/reports`, `/alerts`, `/documents/stats`, монтирование `/static`); оба графа LangGraph компилируются; наличие таблиц БД и `CATEGORIES`. |
| `tests/test_app_running.py` | «приложение запущено» — офлайн через `TestClient` (startup замокан): `/health` = ok, `/` отдаёт HTML веб-чата, `/info` = JSON; плюс e2e-вариант по HTTP. |
| `tests/test_llm.py` | «запросы к LLM проходят» — юнит на моках (`chat`/`embed`/`embed_one`) + реальный вызов под маркером `llm` (skip без доступного провайдера). |
| `tests/test_seed.py` | «для SEED собрана статистика» — юнит: оркестрация `run_seed()` (моки) возвращает сводку с компаниями/отчётами/статистикой; интеграция (e2e): `POST /collect`, затем `GET /documents/stats` и `/reports`. |
| `tests/test_analytics_queries.py` | «≥3 запроса на аналитические сводки» (заданы 4 запроса на русском) + «сводки объективны»: юнит-прогон графа запроса на моках с проверкой, что ответ непустой и опирается на источники; эвристическая проверка объективности; интеграционная проверка объективности через LLM-судью (strict JSON, порог `score ≥ 0.6`, либо честный ответ «недостаточно данных»). |

### Маркеры pytest

| Маркер | Когда применяется | Что требуется |
|---|---|---|
| `integration` | Тесты, работающие против реального стека. | Запущенный стек `app`/БД. |
| `llm` | Тесты, делающие реальный вызов LLM. | Доступный LLM-провайдер/ключ. |
| `e2e` | Полный HTTP end-to-end. | Запущенное приложение по `APP_TEST_BASE_URL`. |

Тесты без этих маркеров — офлайн-юнит на моках.

### Команды запуска

**Только офлайн юнит-тесты** (без сервисов):

```bash
pytest -m "not integration and not llm and not e2e"
```

**Полный прогон** (ресурсо-зависимые тесты сами пропускаются, если ресурс недоступен):

```bash
pytest
```

**В Docker против поднятого стека** (одноразовый сервис под профилем `test`):

```bash
docker compose --profile test run --rm tests
```

**Внутри уже работающего контейнера приложения:**

```bash
docker compose exec app pytest
```

Для `integration`/`e2e`-тестов нужен запущенный стек (`docker compose up -d`), а для `llm`-тестов — валидный `CHAT_PROVIDER`/`OPENAI_API_KEY` или запущенный Ollama. Базовый URL задаётся через `APP_TEST_BASE_URL` (в docker-сервисе `tests` он равен `http://app:8000`).

> Четыре примера аналитических запросов на русском зашиты в `tests/test_analytics_queries.py` и используются как для юнит-прогона графа запроса на моках, так и для интеграционной проверки объективности сводок.

---

## Кому подойдёт этот агент

> Ниже — продуктовые гипотезы и предположения о пользователях. Это MVP, и сегменты сформулированы как гипотезы ценности, требующие проверки.

- **Продуктовые и маркетинговые аналитики.** Гипотеза: им нужна регулярная сводка активности конкурентов (новости, релизы, ценовые движения) без ручного мониторинга десятков источников. Ценность — экономия часов рутины и единая лента изменений.
- **Отделы конкурентной разведки (CI / market intelligence).** Гипотеза: ценят непрерывный автоматический сбор + хранилище с историей и возможность задавать аналитические вопросы на естественном языке со ссылками на первоисточники.
- **Продакт-менеджеры.** Гипотеза: используют алерты (изменение цен, всплеск новостей, найм) как ранние сигналы о шагах конкурентов для приоритизации roadmap.
- **Стартапы, отслеживающие крупных игроков.** Гипотеза: небольшой команде важно дешево и автоматически следить за лидерами рынка (в демо — Apple и Microsoft), не нанимая отдельного аналитика.

**Предположения о пользователях:**

- Пользователь технически способен запустить Docker Compose и заполнить `.env`.
- У пользователя есть доступ к LLM-провайдеру (бюджет на токены).
- Пользователь принимает, что данные собираются из открытого веба и носят справочный характер (нужна верификация по ссылкам).
- Список конкурентов относительно стабилен и задаётся конфигурацией.

---

## Edge cases и обработка ошибок

Система спроектирована «defensive»: внешние сбои не должны останавливать конвейер. Ниже — ключевые граничные случаи и фактическая обработка.

1. **Неполные данные / пустые результаты.**
   - По компании/категории может не найтись документов. `dedup_node` отбрасывает документы без `title` и `content`. Если по компании нет данных, `generate_report` формирует отчёт с честной формулировкой «не найдено данных… недостаточно информации» и `doc_count = 0`.
   - В Q&A при отсутствии релевантных документов `answer_node` возвращает «Недостаточно данных для ответа…», не вызывая LLM с пустым контекстом.

2. **Ошибки внешних API (SearXNG / LLM / Langfuse).**
   - **SearXNG:** `SearxngClient.search()` использует ретраи и при ошибке возвращает `[]` — поиск никогда не «роняет» граф. Дополнительно `collect_node` оборачивает каждый поиск в try/except и копит сообщения в `errors`.
   - **LLM (chat):** вызовы защищены ретраями (tenacity). При сбое генерации отчёта `generate_report` возвращает минимальный отчёт; в `answer_node` — сообщение «Не удалось сформировать ответ из-за ошибки LLM».
   - **LLM (embeddings):** при сбое батч-эмбеддинга `RAGStore.add_documents` вставляет документы с `embedding=None` (данные не теряются). При сбое эмбеддинга запроса `similarity_search` делает **фолбэк на `recent_documents`** (свежие документы).
   - **Langfuse:** при отсутствии ключей или недоступности сервиса трассировка работает в режиме **no-op** (`app/tracing.py`), не влияя на запросы.
   - **БД:** `init_db` выполняет коннект с ретраями; персист отчётов/алертов обёрнут так, что не «роняет» пайплайн.

3. **Конфликтующие или неоднозначные запросы пользователя.**
   - Запрос на естественном языке может быть размыт. Семантический `similarity_search` по эмбеддингам возвращает наиболее близкие по смыслу документы (даже при неточных формулировках), а LLM-аналитик отвечает **только по предоставленному контексту** и **обязан ссылаться на источники**, честно сообщая о недостатке данных.
   - Пустой запрос отклоняется на уровне API (`400`).
   - Фильтры `company` и `period_days` позволяют разрешить неоднозначность (сузить компанию и временное окно).

4. **Ограничения по времени/объёму ответа.**
   - Поиск ограничен `max_results = 8` на запрос (`collect_node`), чтобы сбор был быстрым.
   - В промпт отчёта подаётся не более `40` документов, по `600` символов каждый; в Q&A извлекается до `10` документов, сниппеты обрезаются до `500` символов — это ограничивает латентность и стоимость LLM.
   - Эндпоинты `/reports` и `/alerts` ограничены (`limit` 50/100).

5. **Дубликаты.**
   - **Двухуровневая дедупликация:** в батче и относительно БД по `content_hash` (точное совпадение). `RAGStore.add_documents` пропускает уже существующие хэши и считает `duplicates`.
   - Дополнительно есть утилита `is_near_duplicate()` (косинусная близость) для оценки близких (не точных) дубликатов.

---

## Почему недостаточно обычного детерминированного пайплайна

Жёсткий детерминированный пайплайн (фиксированные правила, regex, парсеры под конкретные сайты) плохо справляется с задачей конкурентной разведки на открытом вебе по нескольким причинам:

- **Неструктурированные и шумные данные.** Результаты метапоиска — это разнородные тексты с разных доменов, без единой схемы. Правила «под сайт» хрупки и быстро ломаются при изменении источников.
- **Необходимость семантического понимания и суммаризации.** Из десятков фрагментов нужно построить связную аналитическую сводку и выделить тренды — это требует LLM, а не шаблонной агрегации.
- **Неоднозначные запросы на естественном языке.** Пользователь спрашивает «что нового у Microsoft с ценами?» — детерминированный поиск по ключевым словам не уловит смысл и синонимы; нужен **семантический RAG-поиск** по эмбеддингам.
- **Изменчивость источников и форматов.** Состав выдачи и структура страниц постоянно меняются; система должна адаптироваться, а не полагаться на фиксированные селекторы.
- **Адаптивные выводы и тренды.** Тренды/алерты зависят от содержания и динамики данных, а не от заранее зашитых условий — эвристики дополняются LLM-анализом.
- **Динамическое ветвление и graceful degradation.** При сбоях (нет результатов, упал эмбеддинг, недоступен LLM) система должна деградировать корректно (фолбэк на свежие документы, частичный отчёт, накопление `errors`), а не падать. Оркестрация графом делает это управляемым и наблюдаемым.

Иными словами, вариативность формулировок, неполнота и шум данных делают детерминированный пайплайн недостаточным: нужна комбинация **RAG + LLM + устойчивая оркестрация графом**.

---

## Критерии оценки результативности

Ниже — предлагаемые метрики и приемлемые пороги для MVP с обоснованием. Пороги подобраны как разумный баланс между полезностью продукта и реалистичностью работы на открытых веб-данных.

| Критерий | Метрика | Порог | Обоснование |
|---|---|---|---|
| **Релевантность/точность ответов** | Доля ответов `/query`, содержащих корректные ссылки на источники, релевантные вопросу | **≥ 80%** | Ответы со ссылками верифицируемы; 80% — практичный порог доверия для аналитика, оставляющий место на шум выдачи. |
| **Полнота сбора** | Доля категорий с данными по компании (из 5) | **≥ 4 из 5** | Открытый веб не всегда даёт данные по всем категориям (особенно `price`); 4/5 означает охват большинства аспектов. |
| **Скорость ответа `/query`** | p95 времени ответа | **≤ 10–15 c** | Включает эмбеддинг запроса, поиск по pgvector и генерацию LLM; верхняя граница ориентирована на медленные модели/сети. |
| **Доля успешных сборов** | Сборы без критических ошибок (пустой `errors` или только некритичные) | **≥ 90%** | Сбор устойчив к сбоям источников; 90% — приемлемая надёжность для фонового процесса. |
| **Уровень ложных алертов** | Доля алертов, признанных нерелевантными при ручной проверке | **≤ 15%** | Алерты основаны на простых эвристиках; ≤15% сохраняет доверие к сигналам, не перегружая пользователя. |

> Метрики ориентировочные для MVP и должны уточняться на реальных данных и сценариях использования.

---

## Источники данных и интеграции

| Интеграция | Тип / формат | Назначение |
|---|---|---|
| **SearXNG (метапоиск)** | HTTP, JSON API | Единая точка доступа к открытому вебу: новости, сайты, отзывы, упоминания цен, вакансии. Запросы строятся по шаблонам категорий (`app/search/queries.py`). |
| **SearXNG MCP-инструмент** | Внутренняя абстракция поиска | `SearxngMCPClient` / `get_search_tool()` — MCP-совместимый поисковый инструмент над SearXNG, который вызывают узлы графа (развязывает узлы от деталей поиска). |
| **LLM-провайдер** | OpenAI-совместимый API (chat + embeddings) | `chat()` — генерация отчётов и ответов на русском; `embed()` / `embed_one()` — построение векторов для RAG. Поддерживаются **OpenAI** и **Ollama** (локальный, OpenAI-совместимый), выбираются независимо для чата и эмбеддингов через `CHAT_PROVIDER` / `EMBEDDING_PROVIDER` в `.env`. |
| **PostgreSQL + pgvector** | СУБД + векторное расширение | Хранилище RAG: документы с эмбеддингами и поиск по косинусной близости. Таблицы: `documents`, `reports`, `alerts`, `collection_runs`. |
| **Langfuse** | HTTP, SDK | Трассировка пользовательских запросов `/query` и ключевых операций (`@traced`). Опционально; no-op при недоступности. |

---

## Ограничения MVP и возможные улучшения

**Ограничения:**

- Компании по умолчанию — только Apple и Microsoft (задаются через `COMPANIES`).
- Сбор хранит результаты поиска (title + сниппет), без полнотекстового скрапинга страниц.
- Дедупликация в проде — точная по `content_hash`; near-duplicate утилита не встроена в основной поток записи.
- Алерты построены на простых эвристиках (без ML-моделей и порогов, обученных на истории).
- Категория документа определяется поисковым запросом, а не классификатором содержимого.
- Графы линейные (без условного ветвления и циклов уточнения).
- Веб-чат минималистичный: без истории сессий и аутентификации.

**Возможные улучшения:**

- Полнотекстовый скрапинг и извлечение фактов из страниц (bs4/lxml уже в зависимостях).
- Near-duplicate дедупликация и кластеризация по смыслу в основном потоке.
- Условное ветвление в графах (уточнение запроса, повторный поиск при нехватке данных).
- ML-классификация категорий и более «умные» алерты (детекция аномалий, тренд-анализ).
- Уведомления (email/Slack) по алертам и e-mail-дайджесты отчётов.
- Расширение списка компаний и источников, многоязычная выдача.
- Кэширование эмбеддингов и батчинг для снижения латентности/стоимости.

---

## Лицензия и дисклеймер

MVP предоставляется «как есть», в демонстрационных целях. Данные собираются из открытых веб-источников через метапоиск и могут быть неполными или неточными — перед принятием решений проверяйте первоисточники по приведённым ссылкам. Соблюдайте условия использования внешних сервисов (SearXNG-инстанса, LLM-провайдера) и применимое законодательство. Лицензия не указана — уточните условия использования у владельца репозитория.
