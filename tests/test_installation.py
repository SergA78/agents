"""Проверка: «приложение установлено».

Verifies that all required third-party dependencies are importable, that the
``app`` package and its key submodules import, and that the expected delivery
files are present on disk.
"""

from __future__ import annotations

import importlib

import pytest

from tests.conftest import PROJECT_ROOT

# Third-party runtime dependencies the MVP relies on. Each entry is the
# *import* name (which can differ from the pip distribution name).
REQUIRED_THIRD_PARTY = [
    "langgraph",
    "langchain",
    "langchain_core",
    "langchain_openai",
    "openai",
    "fastapi",
    "uvicorn",
    "pydantic",
    "pydantic_settings",
    "sqlalchemy",
    "psycopg",
    "pgvector",
    "apscheduler",
    "langfuse",
    "httpx",
    "bs4",  # beautifulsoup4
    "lxml",
    "tenacity",
    "numpy",
]

# Key app submodules that must import cleanly.
APP_SUBMODULES = [
    "app",
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

# Required delivery files (relative to the project root).
REQUIRED_FILES = [
    "docker-compose.yml",
    "Dockerfile",
    "requirements.txt",
    ".env.example",
    "README.md",
    "pytest.ini",
    "app/static/index.html",
    "app/static/styles.css",
    "app/static/app.js",
]


@pytest.mark.parametrize("module_name", REQUIRED_THIRD_PARTY)
def test_third_party_dependency_importable(module_name):
    """Every required third-party dependency imports without error."""
    try:
        importlib.import_module(module_name)
    except ImportError as exc:  # pragma: no cover - failure path
        pytest.fail(f"Required dependency '{module_name}' is not installed: {exc}")


@pytest.mark.parametrize("module_name", APP_SUBMODULES)
def test_app_module_importable(module_name):
    """The app package and each key submodule import cleanly."""
    try:
        importlib.import_module(module_name)
    except ImportError as exc:  # pragma: no cover - failure path
        pytest.fail(f"App module '{module_name}' failed to import: {exc}")


@pytest.mark.parametrize("rel_path", REQUIRED_FILES)
def test_required_delivery_file_exists(rel_path):
    """Each required delivery artifact exists on disk."""
    path = PROJECT_ROOT / rel_path
    assert path.exists(), f"Required file is missing: {path}"
