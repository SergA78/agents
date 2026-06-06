"""SQLAlchemy 2.0 ORM models for the Competitor Intelligence Agent.

Embedding columns use pgvector's ``Vector`` type sized to the configured
embedding dimension.
"""

from __future__ import annotations

from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.config import settings
from app.db.session import Base


class Document(Base):
    """A single collected competitor data item (news, review, price, etc.)."""

    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    company: Mapped[str] = mapped_column(String, index=True)
    category: Mapped[str] = mapped_column(String, index=True)
    title: Mapped[str] = mapped_column(String)
    url: Mapped[str] = mapped_column(String)
    content: Mapped[str] = mapped_column(Text)
    content_hash: Mapped[str] = mapped_column(
        String, unique=True, index=True, nullable=False
    )
    source: Mapped[str | None] = mapped_column(String, nullable=True)
    published_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    collected_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow
    )
    # Stored under the column name "metadata"; "metadata" is reserved on the
    # declarative base, so the attribute is named ``doc_metadata``.
    doc_metadata: Mapped[dict | None] = mapped_column(
        "metadata", JSON, nullable=True
    )
    embedding: Mapped[list[float] | None] = mapped_column(
        Vector(settings.embedding_dim), nullable=True
    )


class Report(Base):
    """An analytical summary report for a company over a time period."""

    __tablename__ = "reports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    company: Mapped[str] = mapped_column(String, index=True)
    period_days: Mapped[int] = mapped_column(Integer)
    summary: Mapped[str] = mapped_column(Text)
    trends: Mapped[dict | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow
    )


class Alert(Base):
    """A notable event detected for a company."""

    __tablename__ = "alerts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    company: Mapped[str] = mapped_column(String, index=True)
    category: Mapped[str] = mapped_column(String)
    alert_type: Mapped[str] = mapped_column(String)
    severity: Mapped[str] = mapped_column(String)
    message: Mapped[str] = mapped_column(Text)
    document_id: Mapped[int | None] = mapped_column(
        ForeignKey("documents.id"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow
    )


class CollectionRun(Base):
    """A record of a single data collection run and its outcome."""

    __tablename__ = "collection_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    started_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    status: Mapped[str] = mapped_column(String)
    stats: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
