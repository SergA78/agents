"""LangGraph graph construction and high-level run helpers.

Two graphs:

* Collection graph: ``collect -> dedup -> store -> report -> alert`` (linear).
* Query graph: ``retrieve -> answer`` (linear).

Graphs are compiled lazily and cached at module level so import stays cheap
and no heavy work happens at import time.
"""

from __future__ import annotations

import logging

from langgraph.graph import END, START, StateGraph

from app.config import CATEGORIES, settings
from app.graph import nodes
from app.graph.state import CollectionState, QueryState
from app.tracing import traced

logger = logging.getLogger(__name__)

# Module-level compiled-graph caches.
_collection_graph = None
_query_graph = None


def build_collection_graph():
    """Build and compile the collection StateGraph."""
    graph = StateGraph(CollectionState)
    graph.add_node("collect", nodes.collect_node)
    graph.add_node("dedup", nodes.dedup_node)
    graph.add_node("store", nodes.store_node)
    graph.add_node("report", nodes.report_node)
    graph.add_node("alert", nodes.alert_node)

    graph.add_edge(START, "collect")
    graph.add_edge("collect", "dedup")
    graph.add_edge("dedup", "store")
    graph.add_edge("store", "report")
    graph.add_edge("report", "alert")
    graph.add_edge("alert", END)

    return graph.compile()


def build_query_graph():
    """Build and compile the query StateGraph."""
    graph = StateGraph(QueryState)
    graph.add_node("retrieve", nodes.retrieve_node)
    graph.add_node("answer", nodes.answer_node)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "answer")
    graph.add_edge("answer", END)

    return graph.compile()


def get_collection_graph():
    """Return the cached compiled collection graph (compiling on first use)."""
    global _collection_graph
    if _collection_graph is None:
        _collection_graph = build_collection_graph()
    return _collection_graph


def get_query_graph():
    """Return the cached compiled query graph (compiling on first use)."""
    global _query_graph
    if _query_graph is None:
        _query_graph = build_query_graph()
    return _query_graph


@traced("run_collection")
def run_collection(
    companies: list[str] | None = None,
    categories: list[str] | None = None,
    time_range: str = "month",
) -> dict:
    """Invoke the collection graph and return the final state.

    Args:
        companies: Companies to collect for; defaults to ``settings.companies_list``.
        categories: Categories to collect; defaults to all ``CATEGORIES``.
        time_range: SearXNG recency window (e.g. ``"day"``, ``"week"``, ``"month"``).

    Returns:
        The final merged graph state dict (includes ``store_stats``, ``report``,
        ``alerts``, ``errors``, etc.).
    """
    init_state: dict = {
        "companies": companies or settings.companies_list,
        "categories": categories or list(CATEGORIES),
        "time_range": time_range,
        "raw_results": [],
        "deduped": [],
        "errors": [],
    }
    logger.info(
        "run_collection: companies=%s categories=%s time_range=%s",
        init_state["companies"], init_state["categories"], time_range,
    )
    graph = get_collection_graph()
    final_state = graph.invoke(init_state)
    return final_state


@traced("run_query")
def run_query(
    query: str,
    company: str | None = None,
    period_days: int | None = None,
) -> dict:
    """Invoke the query graph and return ``{"answer", "retrieved"}``."""
    init_state: dict = {
        "query": query,
        "company": company,
        "errors": [],
    }
    if period_days is not None:
        init_state["period_days"] = period_days

    logger.info(
        "run_query: query=%r company=%s period_days=%s",
        query, company, period_days,
    )
    graph = get_query_graph()
    final_state = graph.invoke(init_state)
    return {
        "answer": final_state.get("answer", ""),
        "retrieved": final_state.get("retrieved", []),
    }
