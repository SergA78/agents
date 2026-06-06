"""Shared pytest fixtures and helpers for the Competitor Intelligence Agent.

Design principles
-----------------
* Plain ``pytest`` must work anywhere. Unit tests run fully offline with mocks.
* Tests that need external resources (a running app, a database, a reachable
  LLM provider) self-skip via an autouse fixture when the resource is missing —
  they never fail or error.
* All connection / reachability probes are performed lazily inside fixtures and
  helpers (never at module import time) so test collection stays import-safe
  even when no services are available.

Markers
-------
* ``integration`` — needs the running app and/or DB stack.
* ``llm``         — needs a reachable LLM provider / API key.
* ``e2e``         — full end-to-end via HTTP against a running app.
"""

from __future__ import annotations

import os
import socket
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlparse

import pytest

# Project root = parent of this tests/ directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Default base URL for a locally running app.
_DEFAULT_BASE_URL = "http://localhost:8000"


# ===================== reachability / availability helpers =====================


def app_base_url() -> str:
    """Return the app base URL from env (APP_TEST_BASE_URL or BASE_URL).

    Falls back to ``http://localhost:8000``.
    """
    return (
        os.environ.get("APP_TEST_BASE_URL")
        or os.environ.get("BASE_URL")
        or _DEFAULT_BASE_URL
    ).rstrip("/")


def app_reachable(base_url: str | None = None, timeout: float = 2.0) -> bool:
    """Return True if ``GET {base}/health`` responds, False on any error."""
    base = (base_url or app_base_url()).rstrip("/")
    try:
        import httpx

        resp = httpx.get(f"{base}/health", timeout=timeout)
        return resp.status_code == 200
    except Exception:
        return False


def _host_port_reachable(url: str, timeout: float = 1.5) -> bool:
    """Return True if a TCP connection to the URL's host:port succeeds."""
    try:
        parsed = urlparse(url)
        host = parsed.hostname
        if not host:
            return False
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False


def llm_available() -> bool:
    """Return True if an LLM provider looks usable; conservative on doubt.

    * If chat provider is ``openai`` and an API key is set -> True.
    * Otherwise, if the resolved provider base URL host:port is reachable
      (e.g. a local Ollama / OpenAI-compatible endpoint) -> True.
    * On any uncertainty or error -> False (so ``llm`` tests skip).
    """
    try:
        from app.config import settings

        provider = settings.chat_provider_norm
        if provider == "openai":
            if settings.openai_api_key.strip():
                # API key present: assume the hosted endpoint is usable.
                return True
            # No key: only usable if pointed at a reachable custom endpoint.
            base = settings.openai_base_url
            if "api.openai.com" in base:
                return False
            return _host_port_reachable(base)
        if provider == "ollama":
            return _host_port_reachable(settings.ollama_base_url)
        # Unknown provider: be conservative.
        return False
    except Exception:
        return False


def db_available() -> bool:
    """Return True if the database accepts a trivial query, else False."""
    try:
        from sqlalchemy import text

        from app.db.session import engine

        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


# ===================== fake LLM clients (unit tests) =====================


class _FakeChatClient:
    """Minimal stand-in for an ``openai.OpenAI`` chat client.

    Records the kwargs of the last ``chat.completions.create`` call and returns
    a deterministic content string.
    """

    def __init__(self, content: str = "FAKE_ANSWER") -> None:
        self._content = content
        self.last_create_kwargs: dict | None = None

        client = self

        class _Completions:
            def create(self, **kwargs):
                client.last_create_kwargs = kwargs
                message = SimpleNamespace(content=client._content)
                choice = SimpleNamespace(message=message)
                return SimpleNamespace(choices=[choice])

        class _Chat:
            completions = _Completions()

        self.chat = _Chat()


class _FakeEmbedClient:
    """Minimal stand-in for an ``openai.OpenAI`` embeddings client."""

    def __init__(self, dim: int = 8) -> None:
        self._dim = dim
        self.last_create_kwargs: dict | None = None

        client = self

        class _Embeddings:
            def create(self, **kwargs):
                client.last_create_kwargs = kwargs
                inputs = kwargs.get("input") or []
                data = [
                    SimpleNamespace(embedding=[0.1] * client._dim)
                    for _ in inputs
                ]
                return SimpleNamespace(data=data)

        self.embeddings = _Embeddings()


@pytest.fixture
def fake_openai(monkeypatch):
    """Monkeypatch app.llm chat/embed clients with fakes for unit tests.

    Returns a ``SimpleNamespace`` exposing ``chat`` and ``embed`` fake clients
    so individual tests can assert on the recorded call kwargs.
    """
    import app.llm as llm

    chat_client = _FakeChatClient(content="FAKE_ANSWER")
    embed_client = _FakeEmbedClient(dim=8)

    monkeypatch.setattr(llm, "get_chat_client", lambda: chat_client)
    monkeypatch.setattr(llm, "get_embed_client", lambda: embed_client)

    return SimpleNamespace(chat=chat_client, embed=embed_client)


# ===================== base_url fixture =====================


@pytest.fixture
def base_url() -> str:
    """The app base URL for integration/e2e tests."""
    return app_base_url()


# ===================== autouse skip gate =====================


@pytest.fixture(autouse=True)
def _skip_when_prereqs_missing(request):
    """Self-skip marked tests when their prerequisites are unavailable.

    * ``integration`` / ``e2e`` -> require the app reachable over HTTP.
    * ``llm``                    -> require a usable LLM provider.

    This keeps a plain ``pytest`` run green: unit tests run, resource-bound
    tests skip cleanly instead of erroring.
    """
    node = request.node

    needs_app = (
        node.get_closest_marker("integration") is not None
        or node.get_closest_marker("e2e") is not None
    )
    needs_llm = node.get_closest_marker("llm") is not None

    if needs_app and not app_reachable():
        pytest.skip(
            f"App not reachable at {app_base_url()} "
            "(set APP_TEST_BASE_URL / start the stack to run integration/e2e tests)."
        )

    if needs_llm and not llm_available():
        pytest.skip(
            "LLM provider not available (no API key / unreachable base URL); "
            "skipping llm-marked test."
        )


# ===================== marker registration fallback =====================


def pytest_configure(config):
    """Register markers programmatically as a fallback to pytest.ini."""
    config.addinivalue_line(
        "markers", "integration: needs the running app and/or DB stack."
    )
    config.addinivalue_line(
        "markers", "llm: needs a reachable LLM provider / API key."
    )
    config.addinivalue_line(
        "markers", "e2e: full end-to-end via HTTP against a running app."
    )
