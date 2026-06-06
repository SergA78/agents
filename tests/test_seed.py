"""Проверка: «для SEED собрана статистика».

UNIT variant (always runs): monkeypatches seed's collaborators so the seed
orchestration runs offline, then asserts the returned summary contains the
configured companies, per-company reports, store stats and an errors list —
proving the seed pipeline collects statistics.

INTEGRATION variant (markers ``integration`` + ``e2e``, self-skips when the app
is unreachable): drives the real /collect + /documents/stats + /reports
pipeline over HTTP to verify collected statistics end-to-end.
"""

from __future__ import annotations

import pytest


# ===================== UNIT (always runs, offline orchestration) =====================


def test_run_seed_collects_statistics(monkeypatch):
    """run_seed() orchestrates collection + per-company reports/alerts offline."""
    import app.seed as seed
    from app.config import settings

    canned_collection = {
        "store_stats": {"received": 4, "inserted": 4, "duplicates": 0},
        "errors": [],
    }

    # Neutralize DB init and external collection.
    monkeypatch.setattr(seed, "init_db", lambda: None)
    monkeypatch.setattr(
        seed, "run_collection", lambda **kwargs: dict(canned_collection)
    )

    # seed imports the reports module as ``generator`` -> patch its functions.
    monkeypatch.setattr(
        seed.generator,
        "generate_report",
        lambda company, period_days=30: {
            "company": company,
            "period_days": period_days,
            "summary": f"report for {company}",
            "trends": [],
            "doc_count": 4,
        },
    )
    monkeypatch.setattr(
        seed.generator,
        "detect_alerts",
        lambda company, period_days=7: [
            {
                "company": company,
                "category": "news",
                "alert_type": "news_spike",
                "severity": "medium",
                "message": "stub",
            }
        ],
    )

    summary = seed.run_seed()

    assert isinstance(summary, dict)

    # Companies come from settings (Apple, Microsoft by default).
    expected_companies = settings.companies_list
    assert summary["companies"] == expected_companies
    assert "Apple" in expected_companies
    assert "Microsoft" in expected_companies

    # Store statistics were captured from the collection step.
    assert summary["store_stats"] == canned_collection["store_stats"]
    assert summary["store_stats"]["inserted"] == 4

    # At least one report per company.
    assert summary["reports"] >= len(expected_companies)
    assert summary["reports"] >= 1

    # Alerts aggregated and an errors list is always present.
    assert summary["alerts"] >= len(expected_companies)
    assert isinstance(summary["errors"], list)


# ===================== INTEGRATION / E2E (needs running app) =====================


@pytest.mark.integration
@pytest.mark.e2e
def test_collection_statistics_pipeline_e2e(base_url):
    """Drive /collect then read /documents/stats and /reports over HTTP."""
    import httpx

    try:
        collect_resp = httpx.post(
            f"{base_url}/collect",
            json={
                "companies": ["Apple"],
                "categories": ["news"],
                "time_range": "month",
            },
            timeout=120.0,
        )
    except httpx.HTTPError as exc:
        pytest.skip(f"/collect request failed (treating as unavailable): {exc}")

    assert collect_resp.status_code == 200, collect_resp.text
    collect_data = collect_resp.json()
    assert "store_stats" in collect_data
    assert "errors" in collect_data

    # Document statistics should be readable and well-formed.
    stats_resp = httpx.get(f"{base_url}/documents/stats", timeout=30.0)
    assert stats_resp.status_code == 200, stats_resp.text
    stats = stats_resp.json()
    assert "total" in stats
    assert "per_company" in stats
    assert isinstance(stats["per_company"], dict)
    assert stats["total"] >= 0
    if stats["total"] > 0:
        assert "Apple" in stats["per_company"]

    # Reports endpoint returns a list-shaped structure for the company.
    reports_resp = httpx.get(
        f"{base_url}/reports", params={"company": "Apple"}, timeout=30.0
    )
    assert reports_resp.status_code == 200, reports_resp.text
    reports = reports_resp.json()
    assert "reports" in reports
    assert isinstance(reports["reports"], list)
