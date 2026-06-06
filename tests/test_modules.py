"""Проверка: «все необходимые модули подключены».

Imports every app module, asserts the FastAPI app exposes the expected routes
and static mount, verifies both LangGraph graphs compile and the run helpers
are callable, and checks core domain constants / DB table registration.
"""

from __future__ import annotations

import importlib

import pytest

APP_MODULES = [
    "app.config",
    "app.llm",
    "app.tracing",
    "app.db.session",
    "app.db.models",
    "app.db.init_db",
    "app.rag.store",
    "app.rag.dedup",
    "app.search.searxng",
    "app.search.queries",
    "app.search.mcp_client",
    "app.graph.state",
    "app.graph.nodes",
    "app.graph.builder",
    "app.graph.scheduler",
    "app.reports.generator",
    "app.seed",
    "app.main",
]

EXPECTED_ROUTES = {
    "/",
    "/info",
    "/health",
    "/collect",
    "/query",
    "/reports",
    "/alerts",
    "/documents/stats",
}


@pytest.mark.parametrize("module_name", APP_MODULES)
def test_all_app_modules_import(module_name):
    """All required app modules are importable (i.e. wired in)."""
    importlib.import_module(module_name)


def test_fastapi_app_exposes_expected_routes():
    """The FastAPI app declares every expected endpoint path."""
    from app.main import app

    paths = {getattr(route, "path", None) for route in app.routes}
    missing = EXPECTED_ROUTES - paths
    assert not missing, f"FastAPI app is missing routes: {sorted(missing)}"


def test_fastapi_app_has_static_mount():
    """A static mount for '/static' exists (serving the web UI)."""
    from app.main import app

    # Mounts appear in app.routes with a path of '/static'.
    mount_paths = {getattr(route, "path", None) for route in app.routes}
    assert "/static" in mount_paths, (
        "Static mount '/static' not found; web UI assets would not be served."
    )


def test_collection_graph_compiles_and_runner_callable():
    """The collection graph compiles and run_collection is callable."""
    from app.graph import builder

    graph = builder.get_collection_graph()
    assert graph is not None
    assert callable(builder.run_collection)

    # LangGraph introspection differs across versions; guard it.
    expected_nodes = {"collect", "dedup", "store", "report", "alert"}
    try:
        node_names = set(graph.get_graph().nodes)
        assert expected_nodes.issubset(node_names), (
            f"Collection graph missing nodes: {expected_nodes - node_names}"
        )
    except AssertionError:
        raise
    except Exception:
        # Introspection API unavailable in this LangGraph version: a successful
        # compile above is sufficient evidence the graph is wired.
        pass


def test_query_graph_compiles_and_runner_callable():
    """The query graph compiles and run_query is callable."""
    from app.graph import builder

    graph = builder.get_query_graph()
    assert graph is not None
    assert callable(builder.run_query)

    expected_nodes = {"retrieve", "answer"}
    try:
        node_names = set(graph.get_graph().nodes)
        assert expected_nodes.issubset(node_names), (
            f"Query graph missing nodes: {expected_nodes - node_names}"
        )
    except AssertionError:
        raise
    except Exception:
        pass


def test_categories_constant_content():
    """CATEGORIES contains the expected competitor data categories."""
    from app.config import CATEGORIES

    assert CATEGORIES == ["news", "press_release", "review", "price", "job"]


def test_orm_metadata_registers_expected_tables():
    """Base.metadata registers all expected tables (models are wired in)."""
    # Importing models registers them on Base.metadata.
    import app.db.models  # noqa: F401
    from app.db.session import Base

    tables = set(Base.metadata.tables.keys())
    expected = {"documents", "reports", "alerts", "collection_runs"}
    missing = expected - tables
    assert not missing, f"ORM metadata missing tables: {sorted(missing)}"
