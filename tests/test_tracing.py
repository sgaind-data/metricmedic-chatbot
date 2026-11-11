"""Tracing tests. No network: Langfuse is disabled or replaced by a dummy handler."""

import pytest

from metricmedic import tracing


@pytest.fixture(autouse=True)
def clear_handler_cache():
    tracing.get_langfuse_handler.cache_clear()
    yield
    tracing.get_langfuse_handler.cache_clear()


def test_tracing_disabled_in_tests():
    assert tracing.tracing_enabled() is False
    assert tracing.build_trace_config(session_id="s1") == {}


def test_tracing_disabled_without_keys(monkeypatch):
    monkeypatch.setenv("LANGFUSE_TRACING_ENABLED", "true")
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    assert tracing.tracing_enabled() is False


def test_trace_config_carries_session_user_tags(monkeypatch):
    dummy = object()
    monkeypatch.setattr(tracing, "get_langfuse_handler", lambda: dummy)

    cfg = tracing.build_trace_config(
        session_id="chat-42", user_id="u1", tags=["api"], metadata={"top_k": 3}
    )

    assert cfg["callbacks"] == [dummy]
    assert cfg["metadata"] == {
        "top_k": 3,
        "langfuse_session_id": "chat-42",
        "langfuse_user_id": "u1",
        "langfuse_tags": ["api"],
    }


def test_merge_configs_keeps_both_callbacks_and_metadata():
    a, b = object(), object()
    merged = tracing.merge_configs(
        {"callbacks": [a], "metadata": {"x": 1}},
        {"callbacks": [b], "metadata": {"y": 2}},
    )
    assert merged["callbacks"] == [a, b]
    assert merged["metadata"] == {"x": 1, "y": 2}


def test_flush_is_noop_when_disabled():
    tracing.flush_traces()  # must not raise or touch the network
