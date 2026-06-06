"""Initial data seeding.

Runs once (e.g. on first startup) to populate the database with an initial
collection over the configured companies and all categories for the last
``settings.seed_days`` days, then generates a report and detects alerts per
company. Fully resilient: failures are logged and processing continues.
"""

from __future__ import annotations

import logging

from app.config import CATEGORIES, settings
from app.db.init_db import init_db
from app.graph.builder import run_collection
from app.reports import generator

logger = logging.getLogger(__name__)


def run_seed() -> dict:
    """Perform initial collection + report/alert generation per company.

    Returns a small summary dict with counts (best-effort).
    """
    logger.info("Seed starting...")

    # Ensure schema exists (idempotent; retries DB connect internally).
    try:
        init_db()
    except Exception:  # noqa: BLE001 - log and continue; collection may still partially work
        logger.exception("Seed: init_db failed")

    companies = settings.companies_list
    summary: dict = {
        "companies": companies,
        "store_stats": {},
        "reports": 0,
        "alerts": 0,
        "errors": [],
    }

    # Initial collection over ~seed_days using the "month" recency window.
    try:
        result = run_collection(
            companies=companies,
            categories=list(CATEGORIES),
            time_range="month",
        )
        summary["store_stats"] = result.get("store_stats", {})
        summary["errors"].extend(result.get("errors") or [])
        logger.info("Seed collection: store_stats=%s", summary["store_stats"])
    except Exception as exc:  # noqa: BLE001
        logger.exception("Seed: collection failed")
        summary["errors"].append(f"seed collection: {exc}")

    # Per-company report + alerts, continuing on error.
    for company in companies:
        try:
            generator.generate_report(company, period_days=settings.seed_days)
            summary["reports"] += 1
        except Exception as exc:  # noqa: BLE001
            logger.exception("Seed: report failed for %s", company)
            summary["errors"].append(f"seed report {company}: {exc}")

        try:
            alerts = generator.detect_alerts(company, period_days=7)
            summary["alerts"] += len(alerts)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Seed: alerts failed for %s", company)
            summary["errors"].append(f"seed alerts {company}: {exc}")

    logger.info(
        "Seed complete: companies=%d reports=%d alerts=%d errors=%d",
        len(companies), summary["reports"], summary["alerts"],
        len(summary["errors"]),
    )
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=settings.log_level.upper())
    run_seed()
