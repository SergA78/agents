"""Проверка: «запросы к LLM проходят».

UNIT variant (always runs): monkeypatches the OpenAI-compatible clients with
fakes and asserts the ``chat`` / ``embed`` / ``embed_one`` helpers behave.

INTEGRATION variant (marker ``llm``, self-skips when no provider is available):
makes tiny real calls against the configured provider.
"""

from __future__ import annotations

import pytest


# ===================== UNIT (always runs, mocked) =====================


def test_chat_returns_fake_content_and_uses_resolved_model(fake_openai):
    """chat() returns the fake content and calls create() with the model."""
    import app.llm as llm
    from app.config import settings

    out = llm.chat([{"role": "user", "content": "hi"}])
    assert out == "FAKE_ANSWER"

    kwargs = fake_openai.chat.last_create_kwargs
    assert kwargs is not None, "chat.completions.create was not called"

    # With provider=openai (default) the resolved model is settings.openai_model.
    expected_model = (
        settings.ollama_model
        if settings.chat_provider_norm == "ollama"
        else settings.openai_model
    )
    assert kwargs.get("model") == expected_model
    assert kwargs.get("messages") == [{"role": "user", "content": "hi"}]


def test_chat_explicit_model_override(fake_openai):
    """An explicit model argument overrides the resolved default."""
    import app.llm as llm

    out = llm.chat([{"role": "user", "content": "hi"}], model="custom-model")
    assert out == "FAKE_ANSWER"
    assert fake_openai.chat.last_create_kwargs.get("model") == "custom-model"


def test_embed_returns_one_vector_per_text(fake_openai):
    """embed() returns one vector per input text."""
    import app.llm as llm

    vectors = llm.embed(["a", "b"])
    assert isinstance(vectors, list)
    assert len(vectors) == 2
    assert all(isinstance(v, list) and v for v in vectors)


def test_embed_empty_returns_empty_without_api_call(fake_openai):
    """embed([]) short-circuits to [] without hitting the client."""
    import app.llm as llm

    assert llm.embed([]) == []
    assert fake_openai.embed.last_create_kwargs is None


def test_embed_one_returns_single_vector(fake_openai):
    """embed_one() returns a single non-empty vector."""
    import app.llm as llm

    vec = llm.embed_one("hello")
    assert isinstance(vec, list)
    assert len(vec) > 0
    assert all(isinstance(x, float) for x in vec)


# ===================== INTEGRATION (needs a real provider) =====================


@pytest.mark.llm
def test_llm_chat_real_roundtrip():
    """A tiny real chat completion returns a non-empty string."""
    import app.llm as llm

    answer = llm.chat(
        [{"role": "user", "content": "Ответь одним словом: тест"}],
        temperature=0.0,
    )
    assert isinstance(answer, str)
    assert answer.strip(), "LLM returned an empty chat completion."


@pytest.mark.llm
def test_llm_embed_real_roundtrip():
    """A tiny real embedding returns a non-empty vector of floats."""
    import app.llm as llm

    vectors = llm.embed(["hello"])
    assert len(vectors) == 1
    vec = vectors[0]
    assert len(vec) > 0
    assert all(isinstance(x, (int, float)) for x in vec)
