"""MCP server tests: a real MCP client talks to the server in memory.

No subprocess, no network, no Gemini (fake LLM).
"""

import json

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.runnables import RunnableLambda
from mcp import Client

from metricmedic import MetricMedicRAG, MetricMedicRetriever
from metricmedic.mcp_server import build_server

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(scope="module")
def core():
    return MetricMedicRetriever()


@pytest.fixture
def server(core):
    fake_llm = FakeListChatModel(responses=["### Likely issue\nFake"])
    rag = MetricMedicRAG(retriever=core, llm=fake_llm)
    return build_server(retriever=core, rag=rag)


def payload(result) -> dict:
    """Tool output as a dict (structured content if present, else the JSON text)."""
    structured = getattr(result, "structured_content", None) or getattr(
        result, "structuredContent", None
    )
    if structured:
        return structured.get("result", structured)
    return json.loads(result.content[0].text)


def is_error(result) -> bool:
    return bool(getattr(result, "is_error", None) or getattr(result, "isError", None))


async def test_lists_both_tools(server):
    async with Client(server) as client:
        tools = await client.list_tools()
    names = {t.name for t in tools.tools}
    assert {"search_troubleshooting_kb", "ask_metricmedic"} <= names


async def test_search_returns_ranked_matches(server):
    async with Client(server) as client:
        result = await client.call_tool(
            "search_troubleshooting_kb",
            {"query": "Why did records disappear after an INNER JOIN?", "top_k": 3},
        )
    data = payload(result)
    assert data["in_scope"] is True
    assert len(data["matches"]) == 3
    assert data["matches"][0]["issue"] == "Inner join dropping records"
    scores = [m["score"] for m in data["matches"]]
    assert scores == sorted(scores, reverse=True)


async def test_search_out_of_scope_returns_no_matches(server):
    async with Client(server) as client:
        result = await client.call_tool(
            "search_troubleshooting_kb", {"query": "How do I bake sourdough bread?"}
        )
    data = payload(result)
    assert data["in_scope"] is False
    assert data["matches"] == []


async def test_ask_returns_grounded_answer(server):
    async with Client(server) as client:
        result = await client.call_tool(
            "ask_metricmedic", {"question": "My dashboard is still showing yesterday's data"}
        )
    data = payload(result)
    assert data["answer"].startswith("### Likely issue")
    assert data["in_scope"] is True
    assert data["used_fallback"] is False
    assert len(data["sources"]) == 3


async def test_ask_out_of_scope_never_calls_the_llm(core):
    def boom(_):
        raise AssertionError("LLM must not be called for out-of-scope questions")

    rag = MetricMedicRAG(retriever=core, llm=RunnableLambda(boom))
    server = build_server(retriever=core, rag=rag)
    async with Client(server) as client:
        result = await client.call_tool("ask_metricmedic", {"question": "Best gaming laptop?"})
    data = payload(result)
    assert data["in_scope"] is False
    assert data["sources"] == []


async def test_invalid_arguments_are_rejected(server):
    async with Client(server) as client:
        result = await client.call_tool(
            "search_troubleshooting_kb", {"query": "join problem", "top_k": 0}
        )
    assert is_error(result)


async def test_issues_resource_lists_all_scenarios(server, core):
    async with Client(server) as client:
        result = await client.read_resource("metricmedic://issues")
    issues = json.loads(result.contents[0].text)
    assert len(issues) == len(core.knowledge_base)
