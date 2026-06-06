"""HTTP client for the self-hosted SearXNG JSON API.

SearXNG runs in docker compose (default ``http://searxng:8080``) and is
configured to return JSON. This module exposes a small, resilient client used
by the collection layer to perform web searches.

Design notes:
    * All network access is wrapped with ``tenacity`` retries (3 attempts with
      exponential backoff) on transport errors and non-200 responses.
    * Searching is intentionally *resilient*: on final failure the client logs
      the error and returns an empty list instead of raising, so that one
      flaky external request never aborts a whole collection run.
"""

from __future__ import annotations

import logging

import httpx
from tenacity import (
    RetryError,
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.config import settings

logger = logging.getLogger(__name__)

# Valid SearXNG ``time_range`` values; anything else is dropped from params.
_VALID_TIME_RANGES = {"day", "week", "month", "year"}

# Default HTTP timeout (seconds) for SearXNG requests.
_TIMEOUT_SECONDS = 15.0


class SearxngClient:
    """Thin HTTP client over the SearXNG JSON search API."""

    def __init__(self, base_url: str | None = None) -> None:
        """Create a client.

        Args:
            base_url: Base URL of the SearXNG instance. Defaults to
                ``settings.searxng_url``.
        """
        self.base_url = (base_url or settings.searxng_url).rstrip("/")

    def search(
        self,
        query: str,
        time_range: str | None = None,
        categories: str = "general",
        max_results: int = 10,
        language: str = "en",
    ) -> list[dict]:
        """Run a search and return a normalized list of result dicts.

        Args:
            query: The search query string.
            time_range: Optional recency filter; one of
                ``day``/``week``/``month``/``year``. Invalid values are ignored.
            categories: SearXNG categories (comma-separated). Defaults to
                ``general``.
            max_results: Maximum number of results to return.
            language: Result language code. Defaults to ``en``.

        Returns:
            A list of dicts shaped as::

                {"title", "url", "content", "engine", "published_date"?}

            ``published_date`` is only present when SearXNG provides it. On any
            error this returns ``[]`` (never raises).
        """
        params: dict[str, str] = {
            "q": query,
            "format": "json",
            "categories": categories,
            "language": language,
        }
        if time_range:
            if time_range in _VALID_TIME_RANGES:
                params["time_range"] = time_range
            else:
                logger.warning(
                    "Ignoring invalid time_range=%r (expected one of %s)",
                    time_range,
                    sorted(_VALID_TIME_RANGES),
                )

        try:
            raw_results = self._fetch(params)
        except RetryError as exc:
            logger.error(
                "SearXNG search failed after retries for query=%r: %s",
                query,
                exc.last_attempt.exception() if exc.last_attempt else exc,
            )
            return []
        except Exception as exc:  # pragma: no cover - defensive catch-all
            logger.error("Unexpected SearXNG search error for query=%r: %s", query, exc)
            return []

        normalized = [self._normalize(item) for item in raw_results]
        return normalized[:max_results]

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(httpx.HTTPError),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def _fetch(self, params: dict[str, str]) -> list[dict]:
        """Perform the HTTP GET and return the raw ``results`` list.

        Raises ``httpx.HTTPError`` on transport errors or non-200 responses so
        that tenacity can retry. Callers must handle ``RetryError``.
        """
        url = f"{self.base_url}/search"
        with httpx.Client(timeout=_TIMEOUT_SECONDS) as client:
            response = client.get(url, params=params)
            response.raise_for_status()
            payload = response.json()
        results = payload.get("results", [])
        return results if isinstance(results, list) else []

    @staticmethod
    def _normalize(item: dict) -> dict:
        """Normalize a raw SearXNG result into the app's canonical shape."""
        normalized: dict = {
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "content": item.get("content", ""),
            "engine": item.get("engine", ""),
        }
        published = item.get("publishedDate") or item.get("published_date")
        if published:
            normalized["published_date"] = published
        return normalized


# Shared default client used by the module-level convenience function.
_default_client: SearxngClient | None = None


def _get_default_client() -> SearxngClient:
    """Return a lazily-initialized shared :class:`SearxngClient`."""
    global _default_client
    if _default_client is None:
        _default_client = SearxngClient()
    return _default_client


def search(
    query: str,
    time_range: str | None = None,
    categories: str = "general",
    max_results: int = 10,
    language: str = "en",
) -> list[dict]:
    """Module-level convenience search using a shared default client.

    See :meth:`SearxngClient.search` for argument and return semantics.
    """
    return _get_default_client().search(
        query,
        time_range=time_range,
        categories=categories,
        max_results=max_results,
        language=language,
    )
