"""LangGraph node implementations.

Every node takes the running state dict and returns a *partial* state dict
containing only the keys it sets. Because LangGraph's default channel reducer
overwrites a key with whatever a node returns, list-valued channels are
returned as the full updated list.

All nodes are resilient: exceptions are caught, recorded into ``state["errors"]``
and the node returns gracefully so a single failure never aborts the graph.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

from app.config import CATEGORIES, settings
from app.rag.dedup import content_hash
from app.rag.store import RAGStore
from app.reports import generator
from app.search.mcp_client import get_search_tool
from app.search.queries import build_queries
from app.tracing import traced

logger = logging.getLogger(__name__)

# Per-query result cap to keep the MVP collection fast.
_MAX_RESULTS_PER_QUERY = 8
# Snippet length stored for retrieved docs in query state.
_SNIPPET_CHARS = 500


def _append_error(state: dict, message: str) -> list[str]:
    """Return the errors list with ``message`` appended (non-mutating copy)."""
    errors = list(state.get("errors") or [])
    errors.append(message)
    return errors


def _parse_published(value) -> datetime | None:
    """Best-effort parse of a published-date string into a datetime.

    Tries ``datetime.fromisoformat`` (handling a trailing ``Z``). Returns
    ``None`` on any failure rather than adding a date-parsing dependency.
    """
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    # Normalize a trailing Z (UTC) which fromisoformat rejects on 3.11.
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
        # Drop tzinfo to match naive DateTime columns used elsewhere.
        if dt.tzinfo is not None:
            dt = dt.replace(tzinfo=None)
        return dt
    except Exception:  # noqa: BLE001 - defensive, no extra deps
        return None


# ===================== COLLECTION nodes =====================


@traced("collect_node")
def collect_node(state: dict) -> dict:
    """Run web searches for every (company, category) and map to doc dicts."""
    companies = state.get("companies") or settings.companies_list
    categories = state.get("categories") or list(CATEGORIES)
    time_range = state.get("time_range", "month")

    raw_results: list[dict] = list(state.get("raw_results") or [])
    errors = list(state.get("errors") or [])

    try:
        search = get_search_tool()
    except Exception as exc:  # noqa: BLE001
        logger.error("collect_node: failed to obtain search tool: %s", exc)
        return {
            "raw_results": raw_results,
            "errors": _append_error(state, f"collect: search tool init failed: {exc}"),
        }

    for company in companies:
        try:
            queries = build_queries(company, categories=categories)
        except Exception as exc:  # noqa: BLE001
            logger.warning("collect_node: build_queries failed for %s: %s", company, exc)
            errors.append(f"collect: build_queries failed for {company}: {exc}")
            continue

        for q in queries:
            category = q.get("category")
            query_text = q.get("query")
            try:
                results = search(
                    query_text,
                    time_range=time_range,
                    max_results=_MAX_RESULTS_PER_QUERY,
                )
            except Exception as exc:  # noqa: BLE001 - search is resilient but guard anyway
                logger.warning(
                    "collect_node: search failed (company=%s category=%s): %s",
                    company, category, exc,
                )
                errors.append(f"collect: search failed {company}/{category}: {exc}")
                continue

            for r in (results or [])[:_MAX_RESULTS_PER_QUERY]:
                try:
                    raw_results.append(
                        {
                            "company": company,
                            "category": category,
                            "title": r.get("title") or "",
                            "url": r.get("url") or "",
                            "content": r.get("content") or "",
                            "source": r.get("engine"),
                            "published_at": _parse_published(
                                r.get("published_date")
                            ),
                            "metadata": {
                                "query": query_text,
                                "engine": r.get("engine"),
                            },
                        }
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.debug("collect_node: result map failed: %s", exc)
                    errors.append(f"collect: result map failed: {exc}")

    logger.info(
        "collect_node: companies=%d categories=%d raw_results=%d",
        len(companies), len(categories), len(raw_results),
    )
    return {
        "companies": companies,
        "categories": categories,
        "time_range": time_range,
        "raw_results": raw_results,
        "errors": errors,
    }


@traced("dedup_node")
def dedup_node(state: dict) -> dict:
    """Drop empty docs and remove in-batch exact duplicates by content_hash."""
    raw_results = state.get("raw_results") or []
    deduped: list[dict] = []
    seen: set[str] = set()
    errors = list(state.get("errors") or [])

    for doc in raw_results:
        try:
            title = doc.get("title") or ""
            content = doc.get("content") or ""
            if not title and not content:
                continue
            h = content_hash(title, content, doc.get("url") or "")
            if h in seen:
                continue
            seen.add(h)
            deduped.append(doc)
        except Exception as exc:  # noqa: BLE001
            logger.debug("dedup_node: skipping doc: %s", exc)
            errors.append(f"dedup: doc skipped: {exc}")

    logger.info(
        "dedup_node: in=%d out=%d", len(raw_results), len(deduped)
    )
    return {"deduped": deduped, "errors": errors}


@traced("store_node")
def store_node(state: dict) -> dict:
    """Persist deduped docs via RAGStore (handles DB dedup + embeddings)."""
    deduped = state.get("deduped") or []
    try:
        stats = RAGStore().add_documents(deduped)
    except Exception as exc:  # noqa: BLE001
        logger.error("store_node: add_documents failed: %s", exc, exc_info=True)
        return {
            "store_stats": {"received": len(deduped), "inserted": 0, "duplicates": 0},
            "errors": _append_error(state, f"store: {exc}"),
        }
    logger.info("store_node: %s", stats)
    return {"store_stats": stats}


@traced("report_node")
def report_node(state: dict) -> dict:
    """Generate and persist a report per company over the deduped/recent docs."""
    companies = state.get("companies") or settings.companies_list
    deduped = state.get("deduped") or []
    errors = list(state.get("errors") or [])

    # Group deduped docs by company so each report can use the fresh batch.
    by_company: dict[str, list[dict]] = {}
    for doc in deduped:
        by_company.setdefault(doc.get("company"), []).append(doc)

    reports: dict[str, dict] = {}
    for company in companies:
        try:
            company_docs = by_company.get(company)
            rep = generator.generate_report(
                company,
                docs=company_docs if company_docs else None,
                period_days=settings.seed_days,
            )
            reports[company] = rep
        except Exception as exc:  # noqa: BLE001
            logger.warning("report_node: failed for %s: %s", company, exc, exc_info=True)
            errors.append(f"report: {company}: {exc}")

    return {"report": reports, "errors": errors}


@traced("alert_node")
def alert_node(state: dict) -> dict:
    """Detect and persist alerts per company."""
    companies = state.get("companies") or settings.companies_list
    errors = list(state.get("errors") or [])
    all_alerts: list[dict] = []

    for company in companies:
        try:
            company_alerts = generator.detect_alerts(company, period_days=7)
            all_alerts.extend(company_alerts)
        except Exception as exc:  # noqa: BLE001
            logger.warning("alert_node: failed for %s: %s", company, exc, exc_info=True)
            errors.append(f"alert: {company}: {exc}")

    logger.info("alert_node: %d alerts", len(all_alerts))
    return {"alerts": all_alerts, "errors": errors}


# ===================== QUERY nodes =====================


@traced("retrieve_node")
def retrieve_node(state: dict) -> dict:
    """Retrieve relevant documents for the user's query via similarity search."""
    query = state.get("query") or ""
    company = state.get("company")
    period_days = state.get("period_days")

    since = None
    if period_days:
        try:
            since = datetime.utcnow() - timedelta(days=int(period_days))
        except Exception:  # noqa: BLE001
            since = None

    retrieved: list[dict] = []
    try:
        docs = RAGStore().similarity_search(
            query, company=company, k=10, since=since
        )
        for d in docs:
            content = getattr(d, "content", "") or ""
            retrieved.append(
                {
                    "id": getattr(d, "id", None),
                    "title": getattr(d, "title", "") or "",
                    "url": getattr(d, "url", "") or "",
                    "company": getattr(d, "company", None),
                    "category": getattr(d, "category", None),
                    "content": content[:_SNIPPET_CHARS],
                    "published_at": (
                        getattr(d, "published_at", None).isoformat()
                        if getattr(d, "published_at", None)
                        else None
                    ),
                }
            )
    except Exception as exc:  # noqa: BLE001
        logger.error("retrieve_node: search failed: %s", exc, exc_info=True)
        return {
            "retrieved": [],
            "errors": _append_error(state, f"retrieve: {exc}"),
        }

    logger.info("retrieve_node: retrieved=%d", len(retrieved))
    return {"retrieved": retrieved}


@traced("answer_node")
def answer_node(state: dict) -> dict:
    """Compose an analytical answer in Russian, citing source URLs."""
    query = state.get("query") or ""
    retrieved = state.get("retrieved") or []

    if not retrieved:
        return {
            "answer": (
                "Недостаточно данных для ответа на запрос. В базе нет "
                "релевантных документов по этой теме."
            )
        }

    context_lines: list[str] = []
    for i, d in enumerate(retrieved, start=1):
        context_lines.append(
            f"[{i}] {d.get('title')} ({d.get('company')}/{d.get('category')})\n"
            f"    URL: {d.get('url')}\n"
            f"    {d.get('content')}"
        )
    context = "\n\n".join(context_lines)

    system = (
        "Ты — аналитик конкурентной разведки. Отвечай на русском языке, "
        "опираясь ТОЛЬКО на предоставленный контекст. Делай аналитические "
        "выводы и обязательно ссылайся на источники (URL) в скобках. Если "
        "данных недостаточно — честно сообщи об этом."
    )
    user = (
        f"Вопрос: {query}\n\n"
        f"Контекст (собранные документы):\n{context}\n\n"
        "Дай содержательный аналитический ответ со ссылками на источники."
    )

    try:
        answer = chat_with_messages(system, user)
    except Exception as exc:  # noqa: BLE001
        logger.error("answer_node: LLM failed: %s", exc, exc_info=True)
        return {
            "answer": "Не удалось сформировать ответ из-за ошибки LLM.",
            "errors": _append_error(state, f"answer: {exc}"),
        }

    return {"answer": answer or "Пустой ответ от модели."}


def chat_with_messages(system: str, user: str) -> str:
    """Thin wrapper around :func:`app.llm.chat` (kept importable/testable)."""
    from app.llm import chat

    return chat(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )
