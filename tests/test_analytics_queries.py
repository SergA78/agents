"""Проверки: «предложи минимум 3 запроса на аналитические сводки» +
«проверь, что сводки объективны».

Provides >= 4 realistic Russian analytical-summary requests and:

* UNIT: drives the query graph (retrieve -> answer) with retrieval and the LLM
  mocked, asserting a grounded, non-empty answer that cites the fake sources.
* OBJECTIVITY UNIT: a lightweight heuristic objectivity proxy over the answers.
* OBJECTIVITY INTEGRATION (LLM-judge): for a running app + available LLM, posts
  each query to /query and asks the model to judge objectivity/groundedness as
  strict JSON, asserting a lenient MVP threshold (or insufficient-data pass).
"""

from __future__ import annotations

import json
import re
from types import SimpleNamespace

import pytest

# ===================== analytical summary requests (>= 4) =====================

ANALYTICAL_QUERIES = [
    "Какие ключевые новости и анонсы были у Apple за последний месяц?",
    "Сравни активность найма (вакансии) Apple и Microsoft за последнее время.",
    "Какие изменения цен или продуктовые анонсы были у Microsoft?",
    "Каковы основные тренды в отзывах пользователей на продукты Apple?",
]

# The explicit "not enough data" message produced by answer_node when retrieval
# is empty; used to treat insufficient-data answers as objective.
_INSUFFICIENT_DATA_MARKER = "Недостаточно данных"

# Subjective / opinion markers that an objective factual summary should avoid.
_BANNED_SUBJECTIVE = [
    "по моему мнению",
    "я думаю",
    "я считаю",
    "наверное",
    "лучший в мире",
    "худший в мире",
]


def _fake_docs(company: str = "Apple") -> list[SimpleNamespace]:
    """Return a small list of fake doc-like objects (attribute access)."""
    return [
        SimpleNamespace(
            id=1,
            title=f"{company} представила новый продукт",
            url="https://example.com/news/1",
            company=company,
            category="news",
            content=f"Компания {company} анонсировала обновление линейки.",
            published_at=None,
        ),
        SimpleNamespace(
            id=2,
            title=f"{company} открыла вакансии",
            url="https://example.com/jobs/2",
            company=company,
            category="job",
            content=f"{company} расширяет команду инженеров.",
            published_at=None,
        ),
    ]


# ===================== objectivity heuristic (lightweight proxy) =====================


def assess_objectivity_heuristic(answer: str, sources: list[str]) -> dict:
    """Lightweight groundedness/objectivity proxy.

    Returns a dict with boolean checks. This is a heuristic proxy (not a full
    factuality evaluation):

    * ``non_empty``   — the answer has content.
    * ``grounded``    — when context/sources exist, the answer cites at least
      one provided source URL, OR it is the explicit insufficient-data message.
    * ``objective``   — the answer avoids obvious subjective/opinion markers.
    """
    text = (answer or "").strip()
    lower = text.lower()

    non_empty = bool(text)
    is_insufficient = _INSUFFICIENT_DATA_MARKER.lower() in lower

    if sources:
        grounded = is_insufficient or any(src and src in text for src in sources)
    else:
        # No context: insufficient-data answer is the correct grounded behavior.
        grounded = is_insufficient or non_empty

    objective = not any(marker in lower for marker in _BANNED_SUBJECTIVE)

    return {
        "non_empty": non_empty,
        "grounded": grounded,
        "objective": objective,
        "insufficient": is_insufficient,
    }


# ===================== UNIT: analytical-summary flow (mocked) =====================


@pytest.fixture
def mocked_query_pipeline(monkeypatch):
    """Patch retrieval + answer LLM so run_query works fully offline.

    Returns the fake source URLs so tests can assert grounding.
    """
    import app.graph.nodes as nodes

    docs = _fake_docs("Apple")
    source_urls = [d.url for d in docs]

    # retrieve_node calls RAGStore().similarity_search(...).
    monkeypatch.setattr(
        nodes.RAGStore,
        "similarity_search",
        lambda self, query, company=None, k=10, since=None: list(docs),
    )

    # answer_node composes its answer via chat_with_messages(system, user);
    # return a deterministic, grounded answer that cites the fake source URLs.
    grounded_answer = (
        "Ключевые наблюдения по собранным данным: компания представила новый "
        f"продукт (источник: {source_urls[0]}) и открыла новые вакансии "
        f"(источник: {source_urls[1]})."
    )
    monkeypatch.setattr(
        nodes, "chat_with_messages", lambda system, user: grounded_answer
    )

    return SimpleNamespace(source_urls=source_urls, answer=grounded_answer)


@pytest.mark.parametrize("query", ANALYTICAL_QUERIES)
def test_analytical_summary_flow_unit(query, mocked_query_pipeline):
    """run_query produces a non-empty answer grounded in the fake sources."""
    from app.graph.builder import run_query

    result = run_query(query)

    answer = result.get("answer", "")
    retrieved = result.get("retrieved", [])

    assert isinstance(answer, str) and answer.strip(), "answer must be non-empty"

    # The retrieved docs reflect the fakes.
    assert len(retrieved) == len(mocked_query_pipeline.source_urls)
    retrieved_urls = {r.get("url") for r in retrieved}
    assert retrieved_urls == set(mocked_query_pipeline.source_urls)

    # The answer cites at least one of the provided source URLs.
    assert any(url in answer for url in mocked_query_pipeline.source_urls)


@pytest.mark.parametrize("query", ANALYTICAL_QUERIES)
def test_analytical_summary_is_objective_heuristic(query, mocked_query_pipeline):
    """The mocked analytical answers pass the objectivity heuristic proxy."""
    from app.graph.builder import run_query

    result = run_query(query)
    answer = result.get("answer", "")
    sources = [r.get("url") for r in result.get("retrieved", []) if r.get("url")]

    assessment = assess_objectivity_heuristic(answer, sources)
    assert assessment["non_empty"], "answer must be non-empty"
    assert assessment["grounded"], (
        "answer must cite a provided source or be the insufficient-data message"
    )
    assert assessment["objective"], (
        "answer must avoid obvious subjective/opinion markers"
    )


def test_answer_node_insufficient_data_is_objective():
    """An empty retrieval yields the explicit insufficient-data message.

    Such an answer is treated as objective (it honestly reports missing data).
    """
    from app.graph.nodes import answer_node

    out = answer_node({"query": "что нового?", "retrieved": []})
    answer = out.get("answer", "")
    assert _INSUFFICIENT_DATA_MARKER in answer

    assessment = assess_objectivity_heuristic(answer, sources=[])
    assert assessment["insufficient"]
    assert assessment["objective"]


# ===================== INTEGRATION: LLM-judge objectivity =====================


def _extract_json(text: str) -> dict | None:
    """Best-effort extraction of the first JSON object from model output."""
    if not text:
        return None
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None


def llm_judge_objectivity(query: str, answer: str, sources: list[str]) -> dict:
    """Use the LLM to judge objectivity/groundedness, returning a parsed dict.

    Asks for STRICT JSON ``{"objective", "grounded", "score", "reason"}``. On
    parse failure returns a conservative default that still allows the test to
    proceed (score 0.0) so the assertion logic handles the outcome.
    """
    from app.llm import chat

    system = (
        "Ты — строгий оценщик объективности аналитических сводок. "
        "Ты проверяешь, является ли ответ объективным (без вымысла и "
        "необоснованных мнений) и обоснован ли он приведёнными источниками."
    )
    sources_block = "\n".join(sources) if sources else "(источники не приведены)"
    user = (
        f"Вопрос пользователя:\n{query}\n\n"
        f"Ответ аналитика:\n{answer}\n\n"
        f"Приведённые источники (URL):\n{sources_block}\n\n"
        "Оцени ответ и верни СТРОГО JSON без пояснений вне JSON в формате:\n"
        '{"objective": true/false, "grounded": true/false, '
        '"score": число от 0 до 1, "reason": "краткое обоснование"}'
    )

    raw = chat(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.0,
    )
    parsed = _extract_json(raw)
    if parsed is None:
        return {"objective": False, "grounded": False, "score": 0.0, "reason": raw}
    return parsed


@pytest.mark.llm
@pytest.mark.integration
@pytest.mark.parametrize("query", ANALYTICAL_QUERIES)
def test_analytical_summary_objectivity_llm_judge(query, base_url):
    """Post each analytical query and let the LLM judge objectivity.

    Lenient MVP threshold: score >= 0.6, OR the answer is the explicit
    insufficient-data message (treated as objective). Network errors skip
    rather than fail.
    """
    import httpx

    try:
        resp = httpx.post(
            f"{base_url}/query", json={"query": query}, timeout=120.0
        )
    except httpx.HTTPError as exc:
        pytest.skip(f"/query request failed (treating as unavailable): {exc}")

    assert resp.status_code == 200, resp.text
    data = resp.json()
    answer = data.get("answer", "")
    sources = data.get("sources", []) or []

    assert isinstance(answer, str) and answer.strip(), "answer must be non-empty"

    # Insufficient-data answers are honest/objective by construction.
    if _INSUFFICIENT_DATA_MARKER in answer:
        return

    try:
        verdict = llm_judge_objectivity(query, answer, sources)
    except Exception as exc:
        pytest.skip(f"LLM judge call failed (treating as unavailable): {exc}")

    score = verdict.get("score", 0.0)
    try:
        score = float(score)
    except (TypeError, ValueError):
        score = 0.0

    assert score >= 0.6, (
        f"LLM judge scored the answer non-objective/ungrounded: {verdict}"
    )
