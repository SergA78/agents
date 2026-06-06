"""Report generation and alert detection.

Two public functions:

* :func:`generate_report` — produces an LLM-written analytical summary
  (in Russian) plus extracted trends for a company over a recency window, and
  persists a :class:`~app.db.models.Report` row.
* :func:`detect_alerts` — applies simple, sensible heuristics over recently
  collected documents to surface notable events (price changes, news spikes,
  hiring), persisting :class:`~app.db.models.Alert` rows.

Both are fully resilient: LLM/DB failures are caught and degrade to a minimal
result rather than raising into the graph.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timedelta

from app.db.models import Alert, Report
from app.db.session import get_session
from app.llm import chat
from app.rag.store import RAGStore
from app.tracing import traced

logger = logging.getLogger(__name__)

# Cap how many docs we feed into the LLM prompt to keep latency/cost bounded.
_MAX_DOCS_IN_PROMPT = 40
# Cap content length per doc in the prompt.
_MAX_CONTENT_CHARS = 600


def _doc_to_view(doc) -> dict:
    """Normalize either an ORM Document or a plain dict into a view dict."""
    if isinstance(doc, dict):
        published = doc.get("published_at")
        return {
            "company": doc.get("company"),
            "category": doc.get("category"),
            "title": doc.get("title") or "",
            "url": doc.get("url") or "",
            "content": doc.get("content") or "",
            "published_at": published,
        }
    return {
        "company": getattr(doc, "company", None),
        "category": getattr(doc, "category", None),
        "title": getattr(doc, "title", "") or "",
        "url": getattr(doc, "url", "") or "",
        "content": getattr(doc, "content", "") or "",
        "published_at": getattr(doc, "published_at", None),
    }


def _build_context(views: list[dict]) -> str:
    """Build a category-grouped textual context block for the LLM prompt."""
    by_category: dict[str, list[dict]] = {}
    for v in views[:_MAX_DOCS_IN_PROMPT]:
        by_category.setdefault(v["category"] or "other", []).append(v)

    lines: list[str] = []
    for category, items in by_category.items():
        lines.append(f"### Категория: {category}")
        for v in items:
            content = (v["content"] or "")[:_MAX_CONTENT_CHARS]
            lines.append(f"- Заголовок: {v['title']}")
            if v["url"]:
                lines.append(f"  Источник: {v['url']}")
            if content:
                lines.append(f"  Текст: {content}")
        lines.append("")
    return "\n".join(lines).strip()


def _parse_trends(raw: str) -> tuple[str, list]:
    """Best-effort extraction of a summary and a trends list from LLM output.

    The model is asked to return JSON ``{"summary": "...", "trends": [...]}``.
    If that fails, fall back to treating the whole text as the summary with an
    empty trends list.
    """
    if not raw:
        return "", []

    # Try to locate a JSON object in the response.
    candidate = raw.strip()
    # Strip code fences if present.
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", candidate, re.DOTALL)
    if fence:
        candidate = fence.group(1)
    else:
        brace = re.search(r"\{.*\}", candidate, re.DOTALL)
        if brace:
            candidate = brace.group(0)

    try:
        data = json.loads(candidate)
        summary = str(data.get("summary", "")).strip()
        trends = data.get("trends", [])
        if not isinstance(trends, list):
            trends = [str(trends)]
        if summary:
            return summary, trends
    except Exception:  # noqa: BLE001 - fall back to raw text
        logger.debug("Report JSON parse failed; using raw text as summary.")

    return raw.strip(), []


@traced("generate_report")
def generate_report(
    company: str,
    docs: list[dict] | None = None,
    period_days: int = 30,
) -> dict:
    """Generate and persist an analytical report for ``company``.

    Args:
        company: Company name.
        docs: Optional pre-collected docs (ORM Documents or dicts). If omitted,
            recent documents are fetched from the store.
        period_days: Recency window in days.

    Returns:
        ``{"company", "period_days", "summary", "trends": [...], "doc_count"}``.
        Always returns a dict, even when the LLM call fails.
    """
    since = datetime.utcnow() - timedelta(days=period_days)

    try:
        if docs is None:
            store = RAGStore()
            fetched = store.recent_documents(
                company=company, since=since, limit=_MAX_DOCS_IN_PROMPT
            )
            views = [_doc_to_view(d) for d in fetched]
        else:
            views = [_doc_to_view(d) for d in docs]
    except Exception:  # noqa: BLE001 - never crash on retrieval
        logger.warning(
            "generate_report: failed to gather docs for %s", company,
            exc_info=True,
        )
        views = []

    doc_count = len(views)

    if doc_count == 0:
        summary = (
            f"За последние {period_days} дн. не найдено данных по компании "
            f"{company}. Недостаточно информации для формирования отчёта."
        )
        result = {
            "company": company,
            "period_days": period_days,
            "summary": summary,
            "trends": [],
            "doc_count": 0,
        }
        _persist_report(result)
        return result

    context = _build_context(views)
    system = (
        "Ты — аналитик конкурентной разведки. Ты пишешь сжатые, фактологичные "
        "отчёты на русском языке на основе собранных данных."
    )
    user = (
        f"Проанализируй собранные данные о компании «{company}» за последние "
        f"{period_days} дней. Сформируй аналитический отчёт, охватывающий, где "
        "это возможно: новости, пресс-релизы, отзывы, изменения цен и вакансии. "
        "Выдели ключевые тренды.\n\n"
        "Верни СТРОГО JSON следующего вида (без пояснений вне JSON):\n"
        '{"summary": "связный текст отчёта на русском", '
        '"trends": ["краткий тренд 1", "краткий тренд 2", ...]}\n\n'
        f"Данные:\n{context}"
    )

    summary = ""
    trends: list = []
    try:
        raw = chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
        )
        summary, trends = _parse_trends(raw)
    except Exception:  # noqa: BLE001 - resilient: minimal report on failure
        logger.warning(
            "generate_report: LLM call failed for %s", company, exc_info=True
        )
        summary = (
            f"Не удалось сгенерировать аналитический отчёт по компании "
            f"{company} из-за ошибки LLM. Собрано документов: {doc_count}."
        )
        trends = []

    if not summary:
        summary = (
            f"Отчёт по компании {company}. Собрано документов: {doc_count}."
        )

    result = {
        "company": company,
        "period_days": period_days,
        "summary": summary,
        "trends": trends,
        "doc_count": doc_count,
    }
    _persist_report(result)
    return result


def _persist_report(result: dict) -> None:
    """Persist a Report row; never raises."""
    try:
        with get_session() as session:
            session.add(
                Report(
                    company=result["company"],
                    period_days=result["period_days"],
                    summary=result["summary"],
                    # trends column is JSON; wrap the list so it round-trips.
                    trends={"items": result.get("trends", [])},
                )
            )
    except Exception:  # noqa: BLE001 - persistence must not crash the pipeline
        logger.warning(
            "Failed to persist report for %s", result.get("company"),
            exc_info=True,
        )


@traced("detect_alerts")
def detect_alerts(company: str, period_days: int = 7) -> list[dict]:
    """Detect notable events for ``company`` using simple heuristics.

    Heuristics (all resilient):
      * price: any new "price" document in the window -> high severity
        ``price_change`` alert.
      * job: any new "job" document in the window -> medium severity
        ``hiring`` alert.
      * news: a spike in news volume vs. the immediately preceding window of
        the same length -> medium/high severity ``news_spike`` alert.

    Returns the list of alert dicts and persists matching Alert rows.
    """
    alerts: list[dict] = []
    store = RAGStore()
    now = datetime.utcnow()
    since = now - timedelta(days=period_days)
    prev_since = now - timedelta(days=period_days * 2)

    try:
        recent = store.recent_documents(
            company=company, since=since, limit=500
        )
    except Exception:  # noqa: BLE001
        logger.warning(
            "detect_alerts: failed to fetch recent docs for %s", company,
            exc_info=True,
        )
        recent = []

    # --- price + hiring: per-document alerts -----------------------------
    news_count = 0
    for doc in recent:
        view = _doc_to_view(doc)
        category = view["category"]
        doc_id = getattr(doc, "id", None) if not isinstance(doc, dict) else doc.get("id")
        if category == "price":
            alerts.append(
                {
                    "company": company,
                    "category": "price",
                    "alert_type": "price_change",
                    "severity": "high",
                    "message": (
                        f"Обнаружено изменение/упоминание цены: "
                        f"{view['title']}"
                    ),
                    "document_id": doc_id,
                }
            )
        elif category == "job":
            alerts.append(
                {
                    "company": company,
                    "category": "job",
                    "alert_type": "hiring",
                    "severity": "medium",
                    "message": (
                        f"Новая активность по найму/вакансиям: "
                        f"{view['title']}"
                    ),
                    "document_id": doc_id,
                }
            )
        elif category == "news":
            news_count += 1

    # --- news spike: compare current window to the prior one --------------
    try:
        # count() filters published_at >= since; approximate the previous
        # window by counting docs since prev_since and subtracting current.
        cur_news = sum(
            1 for d in recent if _doc_to_view(d)["category"] == "news"
        )
        prev_window = store.recent_documents(
            company=company, category="news", since=prev_since, limit=500
        )
        # docs in [prev_since, since) approximate the prior window.
        prior_news = 0
        for d in prev_window:
            pub = _doc_to_view(d)["published_at"]
            if pub is not None and prev_since <= pub < since:
                prior_news += 1
        # Spike: current notably exceeds prior (and there is a baseline).
        if cur_news >= 3 and cur_news >= max(prior_news * 2, prior_news + 3):
            severity = "high" if cur_news >= prior_news * 3 else "medium"
            alerts.append(
                {
                    "company": company,
                    "category": "news",
                    "alert_type": "news_spike",
                    "severity": severity,
                    "message": (
                        f"Всплеск новостной активности: {cur_news} новостей "
                        f"за последние {period_days} дн. против {prior_news} "
                        f"в предыдущем периоде."
                    ),
                    "document_id": None,
                }
            )
    except Exception:  # noqa: BLE001
        logger.debug("news_spike heuristic failed", exc_info=True)

    _persist_alerts(alerts)
    # Strip internal document_id from the public return shape per spec.
    return [
        {
            "company": a["company"],
            "category": a["category"],
            "alert_type": a["alert_type"],
            "severity": a["severity"],
            "message": a["message"],
        }
        for a in alerts
    ]


def _persist_alerts(alerts: list[dict]) -> None:
    """Persist Alert rows; never raises."""
    if not alerts:
        return
    try:
        with get_session() as session:
            for a in alerts:
                session.add(
                    Alert(
                        company=a["company"],
                        category=a["category"],
                        alert_type=a["alert_type"],
                        severity=a["severity"],
                        message=a["message"],
                        document_id=a.get("document_id"),
                    )
                )
    except Exception:  # noqa: BLE001
        logger.warning("Failed to persist alerts", exc_info=True)
