"""Document storage and pgvector-backed retrieval for the RAG layer.

``RAGStore`` wraps PostgreSQL/pgvector access for inserting collected
documents (with exact dedup + embeddings) and retrieving them via
nearest-neighbor vector search or recency.

Detached-object strategy
-------------------------
The session factory (``SessionLocal``) is configured with
``expire_on_commit=False`` and ``get_session()`` commits before closing.
Because attributes are not expired on commit, ORM instances returned from a
``get_session()`` block retain their already-loaded column values and can be
read after the session closes without raising ``DetachedInstanceError``. To be
safe we additionally touch every attribute callers rely on (id, company,
category, title, url, content, published_at, collected_at, doc_metadata) while
the session is still open, then ``expunge_all()`` so the objects are cleanly
detached. We never access lazy relationships post-close.
"""

from __future__ import annotations

import logging
from datetime import datetime

from sqlalchemy import func, select

from app.db.models import Document
from app.db.session import get_session
from app.llm import embed, embed_one
from app.rag.dedup import content_hash

logger = logging.getLogger(__name__)


# Attributes eagerly materialized before detaching, so callers can read them
# after the session closes.
_EAGER_ATTRS = (
    "id",
    "company",
    "category",
    "title",
    "url",
    "content",
    "source",
    "published_at",
    "collected_at",
    "doc_metadata",
)


def _materialize(docs: list[Document]) -> list[Document]:
    """Touch needed attributes so detached access is safe, then return them."""
    for doc in docs:
        for attr in _EAGER_ATTRS:
            getattr(doc, attr, None)
    return docs


class RAGStore:
    """Storage + retrieval facade over the ``documents`` table."""

    def add_documents(self, docs: list[dict]) -> dict:
        """Insert documents with exact dedup and embeddings.

        Each dict may contain: company, category, title, url, content,
        source (optional), published_at (optional datetime), metadata
        (optional dict).

        Workflow:
          * Compute ``content_hash`` for each doc.
          * Skip docs whose hash already exists in the DB (exact dedup) and
            skip in-batch duplicates.
          * Embed the (title + content) text of all new docs in a single
            ``embed()`` call.
          * Insert new ``Document`` rows. If embedding fails, insert with
            ``embedding=None`` so collected data is never lost.

        Returns:
            {"received": N, "inserted": M, "duplicates": D}
        """
        received = len(docs)
        if received == 0:
            return {"received": 0, "inserted": 0, "duplicates": 0}

        # Compute hashes and combined embed text up front.
        prepared: list[dict] = []
        for doc in docs:
            title = doc.get("title") or ""
            content = doc.get("content") or ""
            url = doc.get("url") or ""
            h = content_hash(title, content, url)
            prepared.append(
                {
                    "doc": doc,
                    "hash": h,
                    "text": f"{title}\n\n{content}".strip(),
                }
            )

        inserted = 0
        duplicates = 0

        with get_session() as session:
            # Existing hashes already in the DB.
            all_hashes = [p["hash"] for p in prepared]
            existing: set[str] = set()
            if all_hashes:
                rows = session.execute(
                    select(Document.content_hash).where(
                        Document.content_hash.in_(all_hashes)
                    )
                ).all()
                existing = {r[0] for r in rows}

            # Filter to genuinely new docs, dedup within the batch too.
            seen_in_batch: set[str] = set()
            new_items: list[dict] = []
            for item in prepared:
                h = item["hash"]
                if h in existing or h in seen_in_batch:
                    duplicates += 1
                    continue
                seen_in_batch.add(h)
                new_items.append(item)

            # Batch-embed the new docs in a single call.
            embeddings: list[list[float] | None]
            if new_items:
                texts = [item["text"] for item in new_items]
                try:
                    vectors = embed(texts)
                    if len(vectors) != len(new_items):
                        logger.warning(
                            "Embedding count mismatch (got %d, expected %d); "
                            "inserting without embeddings.",
                            len(vectors),
                            len(new_items),
                        )
                        embeddings = [None] * len(new_items)
                    else:
                        embeddings = list(vectors)
                except Exception:  # noqa: BLE001 - robustness: never lose data
                    logger.warning(
                        "Batch embedding failed for %d documents; inserting "
                        "with embedding=None.",
                        len(new_items),
                        exc_info=True,
                    )
                    embeddings = [None] * len(new_items)
            else:
                embeddings = []

            for item, embedding in zip(new_items, embeddings):
                doc = item["doc"]
                row = Document(
                    company=doc.get("company"),
                    category=doc.get("category"),
                    title=doc.get("title") or "",
                    url=doc.get("url") or "",
                    content=doc.get("content") or "",
                    content_hash=item["hash"],
                    source=doc.get("source"),
                    published_at=doc.get("published_at"),
                    doc_metadata=doc.get("metadata"),
                    embedding=embedding,
                )
                session.add(row)
                inserted += 1

        logger.info(
            "add_documents: received=%d inserted=%d duplicates=%d",
            received,
            inserted,
            duplicates,
        )
        return {
            "received": received,
            "inserted": inserted,
            "duplicates": duplicates,
        }

    def similarity_search(
        self,
        query: str,
        company: str | None = None,
        category: str | None = None,
        k: int = 8,
        since: datetime | None = None,
    ) -> list[Document]:
        """Nearest-neighbor search over document embeddings.

        Embeds ``query`` and orders candidate rows by cosine distance on
        ``Document.embedding`` (rows with NULL embeddings are excluded).
        Filters by company/category/published_at>=since when provided. If the
        query embedding fails (or is empty), falls back to the most recent
        documents matching the same filters.
        """
        query_vec: list[float] = []
        try:
            query_vec = embed_one(query)
        except Exception:  # noqa: BLE001 - fall back to recency on failure
            logger.warning(
                "Query embedding failed; falling back to recent_documents.",
                exc_info=True,
            )

        if not query_vec:
            logger.info(
                "No query embedding available; returning recent documents."
            )
            return self.recent_documents(
                company=company,
                category=category,
                since=since,
                limit=k,
            )

        with get_session() as session:
            stmt = select(Document).where(Document.embedding.is_not(None))
            if company is not None:
                stmt = stmt.where(Document.company == company)
            if category is not None:
                stmt = stmt.where(Document.category == category)
            if since is not None:
                stmt = stmt.where(Document.published_at >= since)
            stmt = stmt.order_by(
                Document.embedding.cosine_distance(query_vec)
            ).limit(k)

            results = list(session.execute(stmt).scalars().all())
            _materialize(results)
            session.expunge_all()
            return results

    def recent_documents(
        self,
        company: str | None = None,
        category: str | None = None,
        since: datetime | None = None,
        limit: int = 100,
    ) -> list[Document]:
        """Return recent documents ordered by ``collected_at`` descending."""
        with get_session() as session:
            stmt = select(Document)
            if company is not None:
                stmt = stmt.where(Document.company == company)
            if category is not None:
                stmt = stmt.where(Document.category == category)
            if since is not None:
                stmt = stmt.where(Document.published_at >= since)
            stmt = stmt.order_by(Document.collected_at.desc()).limit(limit)

            results = list(session.execute(stmt).scalars().all())
            _materialize(results)
            session.expunge_all()
            return results

    def count(
        self,
        company: str | None = None,
        since: datetime | None = None,
    ) -> int:
        """Count documents, optionally filtered by company and recency.

        ``since`` filters on ``published_at >= since``.
        """
        with get_session() as session:
            stmt = select(func.count()).select_from(Document)
            if company is not None:
                stmt = stmt.where(Document.company == company)
            if since is not None:
                stmt = stmt.where(Document.published_at >= since)
            return int(session.execute(stmt).scalar_one())
