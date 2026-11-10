"""API tests. No Gemini calls: the app is built with a fake LLM."""

import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.runnables import RunnableLambda

from metricmedic import MetricMedicRAG, MetricMedicRetriever
from metricmedic.api.main import create_app

QUESTION = "Why did records disappear after an INNER JOIN?"


@pytest.fixture(scope="module")
def core_retriever():
    return MetricMedicRetriever()


@pytest.fixture(scope="module")
def client(core_retriever):
    rag = MetricMedicRAG(
        retriever=core_retriever,
        llm=FakeListChatModel(responses=["### Likely issue\nFake answer"]),
    )
    with TestClient(create_app(rag=rag)) as c:  # `with` runs startup/shutdown
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["kb_entries"] > 0


def test_ask_returns_answer_sources_and_metadata(client):
    r = client.post("/ask", json={"question": QUESTION, "top_k": 2})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"].startswith("### Likely issue")
    assert len(body["sources"]) == 2
    assert body["used_fallback"] is False
    assert body["latency_ms"] >= 0
    assert r.headers["X-Request-ID"] == body["request_id"]


def test_request_id_is_propagated(client):
    r = client.post("/ask", json={"question": QUESTION}, headers={"X-Request-ID": "abc123"})
    assert r.headers["X-Request-ID"] == "abc123"
    assert r.json()["request_id"] == "abc123"


@pytest.mark.parametrize(
    "payload",
    [
        {},  # missing question
        {"question": ""},  # empty
        {"question": "   "},  # whitespace only -> InvalidQueryError
        {"question": "x" * 2001},  # too long
        {"question": QUESTION, "top_k": 0},
        {"question": QUESTION, "top_k": 11},
    ],
)
def test_invalid_input_returns_422(client, payload):
    assert client.post("/ask", json=payload).status_code == 422


def test_llm_failure_still_returns_200_with_fallback(core_retriever):
    def boom(_):
        raise RuntimeError("simulated Gemini outage")

    rag = MetricMedicRAG(retriever=core_retriever, llm=RunnableLambda(boom))
    with TestClient(create_app(rag=rag)) as c:
        r = c.post("/ask", json={"question": QUESTION})
    assert r.status_code == 200
    assert r.json()["used_fallback"] is True


def test_unexpected_error_returns_safe_500(core_retriever):
    class BrokenRAG(MetricMedicRAG):
        def answer(self, *args, **kwargs):
            raise RuntimeError("secret internal detail")

    rag = BrokenRAG(retriever=core_retriever, llm=RunnableLambda(lambda _: "unused"))
    with TestClient(create_app(rag=rag), raise_server_exceptions=False) as c:
        r = c.post("/ask", json={"question": QUESTION})
    assert r.status_code == 500
    assert r.json()["detail"] == "Internal server error"
    assert "secret" not in r.text
