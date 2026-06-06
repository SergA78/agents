"""MCP-style search tool over SearXNG.

This module exposes the *search tool* the LangGraph agent calls. It is modeled
as an MCP (Model Context Protocol) tool interface, but for this MVP the
transport is a direct HTTP call to the self-hosted SearXNG instance via
:class:`app.search.searxng.SearxngClient` rather than a full MCP stdio server
process.

Why an adapter instead of a real MCP server?
    Running and supervising a separate MCP server process (stdio transport,
    lifecycle management, JSON-RPC framing) is heavy for an MVP. By depending on
    the abstract :func:`get_search_tool` callable, the rest of the application
    (graph nodes) never touches the concrete transport, so the SearXNG HTTP
    transport can later be swapped for a real MCP server with no changes to
    callers.

Swapping in a real MCP server later:
    A production setup would launch the SearXNG MCP server (for example the
    ``mcp-searxng`` package) over stdio and configure it via environment, e.g.::

        SEARXNG_URL=http://searxng:8080        # SearXNG base URL the MCP server queries
        SEARXNG_MCP_COMMAND=npx -y mcp-searxng # command used to spawn the MCP server

    :class:`SearxngMCPClient` would then connect to that server using an MCP
    client session (initialize handshake, ``tools/list``, ``tools/call``)
    instead of delegating to :class:`SearxngClient`. Because the public
    ``search`` signature and the :func:`get_search_tool` contract stay the same,
    graph nodes remain unchanged.

Resilience:
    As with the underlying HTTP client, search never raises on failure; it
    returns ``[]`` so a single failed lookup cannot abort a collection run.
"""

from __future__ import annotations

import logging
from typing import Callable

from app.search.searxng import SearxngClient

logger = logging.getLogger(__name__)


class SearxngMCPClient:
    """MCP-compatible search tool backed by SearXNG.

    Exposes the same ``search`` interface as :class:`SearxngClient` but is the
    abstraction the LangGraph agent depends on. Internally it delegates to an
    HTTP :class:`SearxngClient` as the transport; this is the seam where a real
    MCP stdio server could be substituted.
    """

    #: Logical tool name exposed to the agent (mirrors an MCP tool name).
    TOOL_NAME = "searxng_search"

    def __init__(self, base_url: str | None = None) -> None:
        """Create the MCP-style client.

        Args:
            base_url: SearXNG base URL forwarded to the HTTP transport.
                Defaults to ``settings.searxng_url`` via :class:`SearxngClient`.
        """
        self._transport = SearxngClient(base_url=base_url)

    def search(
        self,
        query: str,
        time_range: str | None = None,
        categories: str = "general",
        max_results: int = 10,
        language: str = "en",
    ) -> list[dict]:
        """Run a search through the MCP-style tool.

        Mirrors :meth:`app.search.searxng.SearxngClient.search`. Never raises;
        returns ``[]`` on failure.
        """
        try:
            return self._transport.search(
                query,
                time_range=time_range,
                categories=categories,
                max_results=max_results,
                language=language,
            )
        except Exception as exc:  # pragma: no cover - defensive; transport is resilient
            logger.error("MCP search tool failed for query=%r: %s", query, exc)
            return []


# Shared default MCP client backing the exported tool callable.
_default_mcp_client: SearxngMCPClient | None = None


def _get_default_mcp_client() -> SearxngMCPClient:
    """Return a lazily-initialized shared :class:`SearxngMCPClient`."""
    global _default_mcp_client
    if _default_mcp_client is None:
        _default_mcp_client = SearxngMCPClient()
    return _default_mcp_client


def get_search_tool() -> Callable[..., list[dict]]:
    """Return the search callable used by graph nodes.

    The returned callable has the signature::

        (query: str,
         time_range: str | None = None,
         categories: str = "general",
         max_results: int = 10,
         language: str = "en") -> list[dict]

    This is the single abstraction the rest of the app should depend on for web
    search, decoupling callers from the concrete (HTTP / future MCP) transport.
    """
    return _get_default_mcp_client().search
