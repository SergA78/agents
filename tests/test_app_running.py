"""Проверка: «приложение запущено».

UNIT variant (always runs): boots the FastAPI app via ``TestClient`` with the
startup side-effects (init_db / scheduler / seed) monkeypatched to no-ops so
startup is offline and fast, then exercises the public health/UI/info endpoints.

E2E variant (marker ``e2e``, self-skips if the app is unreachable): hits a
running instance over HTTP.
"""

from __future__ import annotations

import pytest


# ===================== UNIT (always runs, offline) =====================


@pytest.fixture
def offline_client(monkeypatch):
    """Yield a TestClient with startup side-effects neutralized.

    ``init_db``, ``start_scheduler`` and ``run_seed`` are imported into
    ``app.main``'s namespace, so they are patched there.
    """
    import app.main as main
    from fastapi.testclient import TestClient

    monkeypatch.setattr(main, "init_db", lambda: None)
    monkeypatch.setattr(main, "start_scheduler", lambda: None)
    monkeypatch.setattr(main, "run_seed", lambda: {})

    with TestClient(main.app) as client:
        yield client


def test_health_endpoint_ok(offline_client):
    """GET /health returns 200 and {"status": "ok"}."""
    resp = offline_client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_root_serves_html_ui(offline_client):
    """GET / returns the HTML web UI."""
    resp = offline_client.get("/")
    assert resp.status_code == 200
    content_type = resp.headers.get("content-type", "")
    assert "text/html" in content_type, f"unexpected content-type: {content_type}"
    body = resp.text
    assert ("Агент конкурентной разведки" in body) or ("<html" in body.lower()), (
        "Root response does not look like the expected HTML UI."
    )


def test_info_endpoint_lists_endpoints(offline_client):
    """GET /info returns JSON containing an 'endpoints' listing."""
    resp = offline_client.get("/info")
    assert resp.status_code == 200
    data = resp.json()
    assert "endpoints" in data
    assert isinstance(data["endpoints"], dict)


# ===================== E2E (needs running app) =====================


@pytest.mark.e2e
def test_e2e_health_ok(base_url):
    """A running instance reports healthy over HTTP."""
    import httpx

    resp = httpx.get(f"{base_url}/health", timeout=5.0)
    assert resp.status_code == 200
    assert resp.json().get("status") == "ok"


@pytest.mark.e2e
def test_e2e_root_reachable(base_url):
    """The running instance serves the root UI over HTTP."""
    import httpx

    resp = httpx.get(f"{base_url}/", timeout=5.0)
    assert resp.status_code == 200
