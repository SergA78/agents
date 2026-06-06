"""Search-query construction for competitor intelligence collection.

Builds one or more search queries per company per category. Templates are kept
data-driven in :data:`QUERY_TEMPLATES` so adding/tuning a category is a
one-line change.
"""

from __future__ import annotations

from app.config import CATEGORIES

# Default recency window applied to generated queries.
DEFAULT_TIME_RANGE = "month"

# Data-driven query templates keyed by category. Each value is a format string
# expecting a single ``{company}`` field.
QUERY_TEMPLATES: dict[str, str] = {
    "news": "{company} news",
    "press_release": "{company} press release announcement",
    "review": "{company} product review customer feedback",
    "price": "{company} price change pricing",
    "job": "{company} careers job openings hiring",
}


def build_queries(
    company: str,
    categories: list[str] | None = None,
) -> list[dict]:
    """Build search queries for a company across the given categories.

    Args:
        company: Competitor/company name (e.g. ``"Apple"``).
        categories: Categories to build queries for. Defaults to the
            application-wide ``CATEGORIES`` from config. Unknown categories are
            skipped.

    Returns:
        A list of dicts shaped as::

            {"company", "category", "query", "time_range"}

        one per resolved category.
    """
    selected = categories if categories is not None else CATEGORIES
    queries: list[dict] = []
    for category in selected:
        template = QUERY_TEMPLATES.get(category)
        if template is None:
            continue
        queries.append(
            {
                "company": company,
                "category": category,
                "query": template.format(company=company),
                "time_range": DEFAULT_TIME_RANGE,
            }
        )
    return queries


def category_from_query(query: str) -> str | None:
    """Best-effort reverse lookup of a category from a built query string.

    Returns the category whose rendered template (with the company stripped)
    most specifically matches the query, or ``None`` if nothing matches. This
    is a convenience helper for logging/debugging and is not relied upon for
    correctness.
    """
    # Compare against the static, company-independent tail of each template
    # (the words after the leading ``{company}`` placeholder).
    best: tuple[int, str] | None = None
    for category, template in QUERY_TEMPLATES.items():
        tail = template.replace("{company}", "").strip().lower()
        if tail and tail in query.lower():
            score = len(tail)
            if best is None or score > best[0]:
                best = (score, category)
    return best[1] if best else None
