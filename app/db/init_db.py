"""Database initialization.

Ensures the pgvector extension exists and creates all ORM tables. Connection
attempts are retried because the database container may still be starting.
Safe to run repeatedly (idempotent).
"""

from __future__ import annotations

import logging

from sqlalchemy import text
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# Importing models registers them on ``Base.metadata`` for create_all.
from app.db import models  # noqa: F401
from app.db.session import Base, engine

logger = logging.getLogger(__name__)


@retry(
    reraise=True,
    stop=stop_after_attempt(10),
    wait=wait_exponential(multiplier=1, min=1, max=15),
    retry=retry_if_exception_type(Exception),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
def _wait_for_db() -> None:
    """Block until a trivial query succeeds against the database."""
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    logger.info("Database connection established.")


def init_db() -> None:
    """Create the pgvector extension and all tables (idempotent)."""
    logger.info("Initializing database...")
    _wait_for_db()

    with engine.begin() as conn:
        logger.info("Ensuring pgvector extension exists...")
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))

    logger.info("Creating tables...")
    Base.metadata.create_all(engine)
    logger.info("Database initialization complete.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    init_db()
