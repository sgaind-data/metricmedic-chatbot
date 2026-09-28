import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from rag import MetricMedicRetriever


def test_retriever_initialization():
    retriever = MetricMedicRetriever()

    assert retriever.knowledge_base
    assert len(retriever.knowledge_base) > 0
    assert retriever.index.ntotal == len(retriever.knowledge_base)


def test_retrieve_returns_results():
    retriever = MetricMedicRetriever()

    results = retriever.retrieve(
        "My dashboard customer count doubled after joining two tables",
        top_k=3,
    )

    assert len(results) == 3
    assert "issue" in results[0]
    assert "score" in results[0]
    assert "response" in results[0]


def test_join_problem_retrieval():
    retriever = MetricMedicRetriever()

    results = retriever.retrieve(
        "My row count exploded after joining two tables",
        top_k=3,
    )

    retrieved_issues = [
        result["issue"].lower()
        for result in results
    ]

    assert any(
        "join" in issue
        or "duplicate" in issue
        or "row explosion" in issue
        for issue in retrieved_issues
    )


def test_stale_dashboard_retrieval():
    retriever = MetricMedicRetriever()

    results = retriever.retrieve(
        "My dashboard is still showing yesterday's data",
        top_k=3,
    )

    retrieved_text = " ".join(
        result["issue"].lower()
        for result in results
    )

    assert (
        "stale" in retrieved_text
        or "refresh" in retrieved_text
        or "late" in retrieved_text
    )


def test_revenue_mismatch_retrieval():
    retriever = MetricMedicRetriever()

    results = retriever.retrieve(
        "Finance says revenue is different from my Power BI dashboard",
        top_k=3,
    )

    retrieved_text = " ".join(
        result["issue"].lower()
        for result in results
    )

    assert (
        "revenue" in retrieved_text
        or "power bi" in retrieved_text
        or "dashboard" in retrieved_text
    )