"""Periodic collection scheduling via APScheduler.

A single :class:`BackgroundScheduler` runs :func:`run_collection` for all
configured companies/categories every ``settings.collect_interval_hours``
hours. Jobs coalesce and are limited to one concurrent instance so overlapping
or backed-up runs do not pile up.
"""

from __future__ import annotations

import logging

from apscheduler.schedulers.background import BackgroundScheduler

from app.config import CATEGORIES, settings
from app.graph.builder import run_collection

logger = logging.getLogger(__name__)

# Job id used to guard against double-scheduling.
_JOB_ID = "periodic_collection"

# Module-level handle so callers can avoid creating two schedulers.
_scheduler: BackgroundScheduler | None = None


def _scheduled_collection() -> None:
    """Job target: collect for all companies/categories. Never raises."""
    try:
        interval = settings.collect_interval_hours
        # Pick a recency window proportionate to the collection cadence.
        if interval <= 24:
            time_range = "day"
        elif interval <= 24 * 7:
            time_range = "week"
        else:
            time_range = "month"
        logger.info("Scheduled collection starting (time_range=%s)", time_range)
        result = run_collection(
            companies=settings.companies_list,
            categories=list(CATEGORIES),
            time_range=time_range,
        )
        logger.info(
            "Scheduled collection finished: store_stats=%s errors=%d",
            result.get("store_stats"),
            len(result.get("errors") or []),
        )
    except Exception:  # noqa: BLE001 - scheduler jobs must never raise
        logger.exception("Scheduled collection failed")


def start_scheduler() -> BackgroundScheduler:
    """Start (or return the already-running) background scheduler.

    Schedules :func:`run_collection` every ``settings.collect_interval_hours``
    hours with coalescing and ``max_instances=1``.
    """
    global _scheduler
    if _scheduler is not None and _scheduler.running:
        logger.info("Scheduler already running; reusing existing instance.")
        return _scheduler

    scheduler = BackgroundScheduler()
    interval_hours = max(1, int(settings.collect_interval_hours))
    scheduler.add_job(
        _scheduled_collection,
        trigger="interval",
        hours=interval_hours,
        id=_JOB_ID,
        replace_existing=True,
        coalesce=True,
        max_instances=1,
    )
    scheduler.start()
    _scheduler = scheduler
    logger.info(
        "Scheduler started: job %r runs every %d hour(s).",
        _JOB_ID, interval_hours,
    )
    return scheduler


def shutdown_scheduler(scheduler: BackgroundScheduler | None = None) -> None:
    """Shut down the scheduler (the given one, or the module-level handle)."""
    global _scheduler
    target = scheduler or _scheduler
    if target is None:
        return
    try:
        if target.running:
            target.shutdown(wait=False)
            logger.info("Scheduler shut down.")
    except Exception:  # noqa: BLE001 - defensive on shutdown
        logger.debug("Scheduler shutdown error", exc_info=True)
    finally:
        if target is _scheduler:
            _scheduler = None
