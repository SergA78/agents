"""FastAPI application for the Competitor Intelligence Agent.

Wires together the orchestration layer:

* On startup: configure logging, initialize the DB, start the periodic
  collection scheduler, and kick off the initial seed in a background thread so
  startup is not blocked.
* On shutdown: stop the scheduler and flush tracing.

Endpoints (see :func:`root`) cover health, on-demand collection, analytical
Q&A, and read access to reports/alerts/document stats.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import func, select

from app import tracing
from app.config import settings
from app.db.init_db import init_db
from app.db.models import Alert, Document, Report
from app.db.session import get_session
from app.graph.builder import run_collection, run_query
from app.graph.scheduler import shutdown_scheduler, start_scheduler
from app.seed import run_seed

logger = logging.getLogger(__name__)

PROJECT_NAME = "Competitor Intelligence Agent"

# Directory holding the static web chat UI assets (index.html, styles.css, app.js).
STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title=PROJECT_NAME)

# Serve the static web UI assets if the folder exists (guarded to avoid crashing
# when the static directory has not been created yet).
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Holds the running scheduler so shutdown can stop it.
_scheduler = None


# ===================== request models =====================


class CollectRequest(BaseModel):
    companies: list[str] | None = None
    categories: list[str] | None = None
    time_range: str | None = "month"


class QueryRequest(BaseModel):
    query: str
    company: str | None = None
    period_days: int | None = None


# ===================== lifecycle =====================


@app.on_event("startup")
def _on_startup() -> None:
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger.info("Starting %s", PROJECT_NAME)

    try:
        init_db()
    except Exception:  # noqa: BLE001 - log; the seed thread will retry init_db
        logger.exception("Startup: init_db failed (will retry during seed)")

    global _scheduler
    try:
        _scheduler = start_scheduler()
    except Exception:  # noqa: BLE001
        logger.exception("Startup: scheduler failed to start")

    # Kick off seed without blocking startup.
    try:
        threading.Thread(target=run_seed, name="seed", daemon=True).start()
        logger.info("Startup: seed thread launched.")
    except Exception:  # noqa: BLE001
        logger.exception("Startup: failed to launch seed thread")


@app.on_event("shutdown")
def _on_shutdown() -> None:
    logger.info("Shutting down %s", PROJECT_NAME)
    try:
        shutdown_scheduler(_scheduler)
    except Exception:  # noqa: BLE001
        logger.debug("Shutdown: scheduler stop failed", exc_info=True)
    try:
        tracing.flush()
    except Exception:  # noqa: BLE001
        logger.debug("Shutdown: tracing flush failed", exc_info=True)


# ===================== helpers =====================


def _iso(value) -> str | None:
    """Serialize a datetime to ISO 8601, or pass through None."""
    if value is None:
        return None
    try:
        return value.isoformat()
    except Exception:  # noqa: BLE001
        return str(value)


# ===================== endpoints =====================


@app.get("/", response_model=None)
def ui():
    """Serve the web chat UI; fall back to the JSON info listing if missing."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return info()


@app.get("/info")
def info() -> dict:
    """Project name and a listing of available endpoints."""
    return {
        "project": PROJECT_NAME,
        "endpoints": {
            "GET /": "Веб-интерфейс чата для аналитических запросов.",
            "GET /info": "Project name and a listing of available endpoints.",
            "GET /health": "Service health check.",
            "POST /collect": "Run a collection synchronously (companies/categories/time_range).",
            "POST /query": "Analytical Q&A over collected data.",
            "GET /reports?company=": "Latest reports (optionally filtered by company).",
            "GET /alerts?company=": "Recent alerts (optionally filtered by company).",
            "GET /documents/stats": "Per-company document counts.",
        },
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/collect")
def collect(req: CollectRequest) -> dict:
    """Run a collection synchronously and return store stats + counts."""
    try:
        result = run_collection(
            companies=req.companies,
            categories=req.categories,
            time_range=req.time_range or "month",
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("/collect failed")
        raise HTTPException(status_code=500, detail=f"Collection failed: {exc}")

    return {
        "store_stats": result.get("store_stats", {}),
        "raw_count": len(result.get("raw_results") or []),
        "deduped_count": len(result.get("deduped") or []),
        "alerts": result.get("alerts", []),
        "errors": result.get("errors", []),
    }


@app.post("/query")
def query(req: QueryRequest) -> dict:
    """Run the analytical query graph (traced via run_query)."""
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")

    try:
        result = run_query(
            req.query,
            company=req.company,
            period_days=req.period_days,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("/query failed")
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}")

    sources = [
        r.get("url")
        for r in (result.get("retrieved") or [])
        if r.get("url")
    ]
    return {
        "answer": result.get("answer", ""),
        "sources": sources,
    }


@app.get("/reports")
def reports(company: str | None = None) -> dict:
    """Return latest reports, optionally filtered by company."""
    try:
        with get_session() as session:
            stmt = select(Report)
            if company:
                stmt = stmt.where(Report.company == company)
            stmt = stmt.order_by(Report.created_at.desc()).limit(50)
            rows = list(session.execute(stmt).scalars().all())
            items = [
                {
                    "id": r.id,
                    "company": r.company,
                    "period_days": r.period_days,
                    "summary": r.summary,
                    "trends": (r.trends or {}).get("items", [])
                    if isinstance(r.trends, dict)
                    else r.trends,
                    "created_at": _iso(r.created_at),
                }
                for r in rows
            ]
    except Exception as exc:  # noqa: BLE001
        logger.exception("/reports failed")
        raise HTTPException(status_code=500, detail=f"Failed to load reports: {exc}")

    return {"count": len(items), "reports": items}


@app.get("/alerts")
def alerts(company: str | None = None) -> dict:
    """Return recent alerts, optionally filtered by company."""
    try:
        with get_session() as session:
            stmt = select(Alert)
            if company:
                stmt = stmt.where(Alert.company == company)
            stmt = stmt.order_by(Alert.created_at.desc()).limit(100)
            rows = list(session.execute(stmt).scalars().all())
            items = [
                {
                    "id": a.id,
                    "company": a.company,
                    "category": a.category,
                    "alert_type": a.alert_type,
                    "severity": a.severity,
                    "message": a.message,
                    "document_id": a.document_id,
                    "created_at": _iso(a.created_at),
                }
                for a in rows
            ]
    except Exception as exc:  # noqa: BLE001
        logger.exception("/alerts failed")
        raise HTTPException(status_code=500, detail=f"Failed to load alerts: {exc}")

    return {"count": len(items), "alerts": items}


@app.get("/documents/stats")
def documents_stats() -> dict:
    """Return per-company document counts plus a total."""
    try:
        with get_session() as session:
            stmt = (
                select(Document.company, func.count())
                .group_by(Document.company)
                .order_by(func.count().desc())
            )
            rows = session.execute(stmt).all()
            per_company = {company: int(count) for company, count in rows}
            total = int(
                session.execute(
                    select(func.count()).select_from(Document)
                ).scalar_one()
            )
    except Exception as exc:  # noqa: BLE001
        logger.exception("/documents/stats failed")
        raise HTTPException(status_code=500, detail=f"Failed to load stats: {exc}")

    return {"total": total, "per_company": per_company}
