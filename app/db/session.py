"""SQLAlchemy 2.0 engine, session factory, and declarative base.

``Base`` is defined here and imported by ``app.db.models``.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import settings


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


# Core engine; ``pool_pre_ping`` guards against stale connections.
engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
    future=True,
)

# Session factory. Sessions do not auto-flush and keep objects usable after
# commit (no expire-on-commit) which is convenient for read-after-write.
SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    expire_on_commit=False,
    future=True,
)


@contextmanager
def get_session() -> Iterator[Session]:
    """Yield a session, committing on success and rolling back on error."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
