"""LangGraph state schemas for the orchestration layer.

These ``TypedDict`` types describe the channels flowing through the two
StateGraphs:

* :class:`CollectionState` — the periodic/seed collection pipeline
  (collect -> dedup -> store -> report -> alert).
* :class:`QueryState` — the on-demand analytical Q&A pipeline
  (retrieve -> answer).

All fields are optional (``total=False``) because each node only sets the
subset of keys it produces; LangGraph merges the returned partial state into
the running state (default channel semantics overwrite a key with the value a
node returns).
"""

from __future__ import annotations

from typing import TypedDict


class CollectionState(TypedDict, total=False):
    """State for the data-collection pipeline."""

    companies: list[str]
    categories: list[str]
    time_range: str
    raw_results: list[dict]
    deduped: list[dict]
    store_stats: dict
    run_id: int
    report: dict
    alerts: list[dict]
    errors: list[str]


class QueryState(TypedDict, total=False):
    """State for the on-demand analytical query pipeline."""

    query: str
    company: str | None
    period_days: int
    retrieved: list[dict]
    answer: str
    errors: list[str]
