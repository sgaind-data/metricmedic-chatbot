"""Chain + pipeline tests. No Gemini calls: a fake chat model stands in for the LLM."""

import logging

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.runnables import RunnableLambda

from metricmedic import InvalidQueryError, MetricMedicRAG, MetricMedicRetriever
from metricmedic.lc_retriever import MetricMedicLCRetriever

QUERY = "My dashboard customer count doubled after joining two tables"


@pytest.fixture(scope="module")
def core_retriever():
    return MetricMedicRetriever()


def _failing_llm():
    def boom(_):
        raise RuntimeError("simulated Gemini outage")

    return RunnableLambda(boom)


def test_lc_retriever_returns_documents(core_retriever):
    lc = MetricMedicLCRetriever(core=core_retriever, top_k=3)
    docs = lc.invoke(QUERY)

    assert len(docs) == 3
    assert all(d.page_content.startswith("Issue:") for d in docs)
    scores = [d.metadata["score"] for d in docs]
    assert scores == sorted(scores, reverse=True)


def test_chain_returns_answer_and_sources(core_retriever):
    fake_llm = FakeListChatModel(responses=["### Likely issue\nFake answer"])
    rag = MetricMedicRAG(retriever=core_retriever, llm=fake_llm)

    result = rag.answer(QUERY, top_k=2)

    assert result["answer"].startswith("### Likely issue")
    assert result["used_fallback"] is False
    assert len(result["sources"]) == 2
    assert {"issue", "keywords", "response", "score"} <= result["sources"][0].keys()


def test_chain_falls_back_when_llm_fails(core_retriever):
    rag = MetricMedicRAG(retriever=core_retriever, llm=_failing_llm())
    result = rag.answer(QUERY)

    assert result["used_fallback"] is True
    assert "The most relevant troubleshooting match is" in result["answer"]
    assert result["sources"][0]["issue"] in result["answer"]


def test_fallback_is_logged_not_silent(core_retriever, caplog):
    """Regression test: the original fallback hid a 400 error with no trace."""
    rag = MetricMedicRAG(retriever=core_retriever, llm=_failing_llm())
    logging.getLogger("metricmedic").propagate = True  # let caplog see it

    with caplog.at_level(logging.WARNING, logger="metricmedic"):
        rag.answer(QUERY)

    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "fallback" in messages
    assert "simulated Gemini outage" in messages


@pytest.mark.parametrize(
    "query, top_k",
    [("", 3), ("   ", 3), ("x" * 2001, 3), (QUERY, 0), (QUERY, 11)],
)
def test_invalid_requests_are_rejected(core_retriever, query, top_k):
    rag = MetricMedicRAG(retriever=core_retriever, llm=_failing_llm())
    with pytest.raises(InvalidQueryError):
        rag.answer(query, top_k=top_k)


def test_prompt_has_no_system_message():
    """Regression test: Gemini v1 rejects system messages (400 INVALID_ARGUMENT)."""
    from metricmedic.prompts import RAG_PROMPT

    messages = RAG_PROMPT.invoke({"question": "q", "context": "c"}).to_messages()
    assert [m.type for m in messages] == ["human"]


def test_embedding_cache_is_reused(tmp_path):
    first = MetricMedicRetriever(cache_dir=tmp_path)
    cached = list(tmp_path.glob("kb_embeddings_*.npy"))
    assert len(cached) == 1

    second = MetricMedicRetriever(cache_dir=tmp_path)
    assert (first.embeddings == second.embeddings).all()
