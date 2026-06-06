"""OpenAI-compatible LLM and embedding integration.

Supports two providers — OpenAI and Ollama — selected independently for chat
and for embeddings via ``settings.chat_provider`` / ``settings.embedding_provider``.
Ollama exposes an OpenAI-compatible API, so the same ``openai`` SDK is reused
for both providers; only the base URL / API key / model name differ.

Provides cached clients plus thin, retry-wrapped helpers for chat completion
and embeddings. All network calls are retried with exponential backoff. The
public helpers (``chat``, ``embed``, ``embed_one``) keep their original
signatures so the rest of the app is unaffected by the provider choice.
"""

from __future__ import annotations

import logging

from openai import OpenAI
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.config import settings

logger = logging.getLogger(__name__)

# Lazily-initialized, per-purpose cached clients. Settings are static at
# runtime, so no invalidation is required.
_chat_client: OpenAI | None = None
_embed_client: OpenAI | None = None


def _resolve(purpose: str) -> tuple[str, str]:
    """Return ``(base_url, api_key)`` for the given purpose.

    Args:
        purpose: Either ``"chat"`` or ``"embed"``.
    """
    provider = (
        settings.chat_provider_norm
        if purpose == "chat"
        else settings.embedding_provider_norm
    )
    if provider == "ollama":
        return settings.ollama_base_url, settings.ollama_api_key
    # Default / "openai" and any unknown value fall back to OpenAI settings.
    return settings.openai_base_url, settings.openai_api_key


def get_chat_client() -> OpenAI:
    """Return a cached OpenAI-compatible client for the chat provider."""
    global _chat_client
    if _chat_client is None:
        base_url, api_key = _resolve("chat")
        _chat_client = OpenAI(api_key=api_key, base_url=base_url)
    return _chat_client


def get_embed_client() -> OpenAI:
    """Return a cached OpenAI-compatible client for the embedding provider."""
    global _embed_client
    if _embed_client is None:
        base_url, api_key = _resolve("embed")
        _embed_client = OpenAI(api_key=api_key, base_url=base_url)
    return _embed_client


def get_client() -> OpenAI:
    """Return the chat client (kept for backward compatibility)."""
    return get_chat_client()


def _chat_model() -> str:
    """Resolve the chat model name based on the configured chat provider."""
    if settings.chat_provider_norm == "ollama":
        return settings.ollama_model
    return settings.openai_model


def _embed_model() -> str:
    """Resolve the embedding model name based on the embedding provider."""
    if settings.embedding_provider_norm == "ollama":
        return settings.ollama_embedding_model
    return settings.embedding_model


# Shared tenacity retry configuration: 3 attempts, exponential backoff,
# log a warning before each sleep/retry.
_retry = retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(Exception),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)


@_retry
def chat(
    messages: list[dict],
    temperature: float = 0.2,
    model: str | None = None,
    **kwargs,
) -> str:
    """Run a chat completion and return the assistant message content.

    Args:
        messages: List of OpenAI-style chat messages.
        temperature: Sampling temperature.
        model: Override model; defaults to ``settings.openai_model``.
        **kwargs: Forwarded to ``chat.completions.create``.
    """
    client = get_chat_client()
    chat_model = model or _chat_model()
    logger.debug("chat via provider=%s model=%s", settings.chat_provider_norm, chat_model)
    response = client.chat.completions.create(
        model=chat_model,
        messages=messages,
        temperature=temperature,
        **kwargs,
    )
    content = response.choices[0].message.content
    return content or ""


@_retry
def embed(texts: list[str]) -> list[list[float]]:
    """Return embedding vectors for the given texts.

    Empty input returns an empty list without calling the API.
    """
    if not texts:
        return []
    client = get_embed_client()
    embed_model = _embed_model()
    logger.debug(
        "embed via provider=%s model=%s", settings.embedding_provider_norm, embed_model
    )
    response = client.embeddings.create(
        model=embed_model,
        input=texts,
    )
    return [item.embedding for item in response.data]


def embed_one(text: str) -> list[float]:
    """Convenience wrapper to embed a single string."""
    vectors = embed([text])
    return vectors[0] if vectors else []
