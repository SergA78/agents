"""Langfuse tracing helpers.

All Langfuse interaction is fully defensive: if Langfuse is not configured or
unreachable, tracing degrades to a no-op and never raises into the caller.
Targets the Langfuse v2 SDK API but guards every call with try/except so it
also tolerates other versions.
"""

from __future__ import annotations

import functools
import logging
from typing import Any, Callable, TypeVar

from app.config import settings

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

# Cached Langfuse client (or None). ``False`` means "not yet initialized".
_langfuse: Any = False


def get_langfuse() -> Any | None:
    """Return a cached Langfuse client, or ``None`` if not configured.

    A client is only created when both the public and secret keys are set.
    Any error during construction degrades gracefully to ``None``.
    """
    global _langfuse
    if _langfuse is not False:
        return _langfuse

    if not (settings.langfuse_public_key and settings.langfuse_secret_key):
        _langfuse = None
        return _langfuse

    try:
        from langfuse import Langfuse

        _langfuse = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Langfuse initialization failed, tracing disabled: %s", exc)
        _langfuse = None

    return _langfuse


def traced(name: str) -> Callable[[F], F]:
    """Decorator that wraps a sync function in a Langfuse trace.

    Records input arguments, output, and exceptions. If Langfuse is
    unavailable the wrapped function is simply called directly. All tracing
    errors are caught and ignored (logged at debug level).
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            client = get_langfuse()
            if client is None:
                return func(*args, **kwargs)

            trace = None
            try:
                trace = client.trace(
                    name=name,
                    input={"args": _safe(args), "kwargs": _safe(kwargs)},
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("Langfuse trace creation failed: %s", exc)
                trace = None

            try:
                result = func(*args, **kwargs)
            except Exception as exc:
                _record_error(trace, exc)
                raise
            else:
                _record_output(trace, result)
                return result

        return wrapper  # type: ignore[return-value]

    return decorator


def flush() -> None:
    """Flush buffered Langfuse events, if a client exists."""
    client = get_langfuse()
    if client is None:
        return
    try:
        client.flush()
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Langfuse flush failed: %s", exc)


# --- internal helpers -----------------------------------------------------


def _safe(value: Any) -> Any:
    """Best-effort conversion of arbitrary values for trace payloads."""
    try:
        return repr(value)
    except Exception:  # pragma: no cover - defensive
        return "<unrepresentable>"


def _record_output(trace: Any, result: Any) -> None:
    if trace is None:
        return
    try:
        trace.update(output=_safe(result))
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Langfuse output update failed: %s", exc)


def _record_error(trace: Any, exc: Exception) -> None:
    if trace is None:
        return
    try:
        trace.update(level="ERROR", status_message=str(exc))
    except Exception as inner:  # pragma: no cover - defensive
        logger.debug("Langfuse error update failed: %s", inner)
