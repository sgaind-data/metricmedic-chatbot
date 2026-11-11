"""Optional Langfuse tracing.

Tracing is ON only when LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set and
LANGFUSE_TRACING_ENABLED is not "false". Without keys the app runs exactly as
before: tracing must never be able to break answering.

Each traced request records: the full chain as a tree of steps, latency per
step, the retrieved documents, the prompt sent to Gemini, the answer, token
counts and an estimated cost.
"""

import logging
import os
from functools import lru_cache

import metricmedic.config  # noqa: F401  (loads .env before we read env vars)

logger = logging.getLogger("metricmedic.tracing")


def tracing_enabled() -> bool:
    if os.getenv("LANGFUSE_TRACING_ENABLED", "true").lower() in ("false", "0", "no"):
        return False
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


@lru_cache(maxsize=1)
def get_langfuse_handler():
    """One shared LangChain callback handler, or None if tracing is off/unavailable."""
    if not tracing_enabled():
        logger.info("Langfuse tracing disabled", extra={"event": "tracing_disabled"})
        return None
    try:
        from langfuse.langchain import CallbackHandler

        handler = CallbackHandler()
        logger.info("Langfuse tracing enabled", extra={"event": "tracing_enabled"})
        return handler
    except Exception as exc:  # never let tracing break the app
        logger.warning(
            "Langfuse tracing unavailable: %s: %s",
            type(exc).__name__,
            exc,
            extra={"event": "tracing_error", "error_type": type(exc).__name__},
        )
        return None


def build_trace_config(
    *,
    session_id: str | None = None,
    user_id: str | None = None,
    tags: list[str] | None = None,
    metadata: dict | None = None,
) -> dict:
    """LangChain `config` for one request. Empty callbacks when tracing is off."""
    handler = get_langfuse_handler()
    if handler is None:
        return {}

    trace_metadata = dict(metadata or {})
    if session_id:
        trace_metadata["langfuse_session_id"] = session_id
    if user_id:
        trace_metadata["langfuse_user_id"] = user_id
    if tags:
        trace_metadata["langfuse_tags"] = tags
    return {"callbacks": [handler], "metadata": trace_metadata}


def merge_configs(base: dict, extra: dict | None) -> dict:
    """Combine our tracing config with a caller-supplied LangChain config."""
    if not extra:
        return base
    merged = {**base, **extra}
    merged["callbacks"] = list(base.get("callbacks", [])) + list(extra.get("callbacks", []))
    merged["metadata"] = {**base.get("metadata", {}), **extra.get("metadata", {})}
    return merged


def flush_traces() -> None:
    """Send buffered traces now. Call at the end of short-lived scripts and on shutdown."""
    if not tracing_enabled():
        return
    try:
        from langfuse import get_client

        get_client().flush()
    except Exception as exc:
        logger.warning("Langfuse flush failed: %s", exc, extra={"event": "tracing_error"})
