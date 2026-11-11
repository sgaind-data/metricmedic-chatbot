"""MetricMedic as an MCP server, so AI agents can use it as a tool.

Run (stdio, for Claude Desktop / Claude Code / MCP Inspector):
    python -m metricmedic.mcp_server
    metricmedic-mcp                       # same, via the console script
Run over HTTP instead:
    python -m metricmedic.mcp_server --transport streamable-http

Tools:
  search_troubleshooting_kb  retrieval only - no LLM call, free and fast. The calling
                             agent is itself an LLM and can reason over the scenarios.
  ask_metricmedic            full grounded RAG answer (1 Gemini call).
Resource:
  metricmedic://issues       titles of all knowledge-base scenarios.

IMPORTANT for stdio: stdout carries the MCP protocol, so this module never prints;
logs go to stderr (see logging_config).
"""

import argparse
import json
import logging
from typing import Annotated

import anyio
from mcp.server import MCPServer
from pydantic import BaseModel, Field

from metricmedic import __version__
from metricmedic.logging_config import setup_logging
from metricmedic.pipeline import MAX_QUERY_CHARS, MAX_TOP_K, MIN_TOP_K, MetricMedicRAG
from metricmedic.retriever import MetricMedicRetriever
from metricmedic.tracing import flush_traces

logger = logging.getLogger("metricmedic.mcp")

# From the Milestone 4 retrieval eval: the highest out-of-scope top-1 score was 0.142
# and the lowest in-scope top-1 score was 0.333. 0.25 sits in the gap.
OUT_OF_SCOPE_THRESHOLD = 0.25

OUT_OF_SCOPE_MESSAGE = (
    "This question does not match any MetricMedic troubleshooting scenario. "
    "MetricMedic covers SQL, dashboard, KPI, reporting, ETL and data-quality issues."
)

INSTRUCTIONS = (
    "MetricMedic diagnoses SQL, dashboard, KPI, reporting, ETL and data-quality problems "
    "using a curated knowledge base of troubleshooting scenarios. Prefer "
    "search_troubleshooting_kb (no LLM cost) and reason over the returned guidance; use "
    "ask_metricmedic when a complete, formatted answer is wanted."
)

Query = Annotated[
    str,
    Field(
        min_length=1,
        max_length=MAX_QUERY_CHARS,
        description="The problem in plain English, e.g. 'my row count doubled after a join'.",
    ),
]
TopK = Annotated[
    int, Field(ge=MIN_TOP_K, le=MAX_TOP_K, description="How many scenarios to retrieve.")
]


class KBMatch(BaseModel):
    issue: str
    guidance: str
    keywords: str
    score: float = Field(description="Cosine similarity, higher is more relevant.")
    kb_index: int


class SearchResult(BaseModel):
    query: str
    in_scope: bool = Field(description="False when nothing in the knowledge base matches.")
    matches: list[KBMatch]
    note: str = ""


class AskResult(BaseModel):
    answer: str
    sources: list[KBMatch]
    used_fallback: bool = Field(description="True if Gemini failed and sources were used.")
    in_scope: bool


def _to_match(source: dict) -> KBMatch:
    return KBMatch(
        issue=source["issue"],
        guidance=source["response"],
        keywords=str(source["keywords"]),
        score=round(source["score"], 4),
        kb_index=source["kb_index"],
    )


def build_server(
    retriever: MetricMedicRetriever | None = None, rag: MetricMedicRAG | None = None
) -> MCPServer:
    """Server factory. Tests inject a retriever and a RAG built with a fake LLM.

    Models are loaded lazily on first use, so the server starts instantly and
    search-only usage never needs a Gemini API key.
    """
    state = {"retriever": retriever, "rag": rag}

    def get_retriever() -> MetricMedicRetriever:
        if state["retriever"] is None:
            state["retriever"] = MetricMedicRetriever()
        return state["retriever"]

    def get_rag() -> MetricMedicRAG:
        if state["rag"] is None:
            state["rag"] = MetricMedicRAG(retriever=get_retriever())
        return state["rag"]

    server = MCPServer(name="metricmedic", instructions=INSTRUCTIONS, version=__version__)

    @server.tool(
        name="search_troubleshooting_kb",
        description=(
            "Search MetricMedic's knowledge base of SQL / dashboard / KPI / ETL / data-quality "
            "troubleshooting scenarios. Returns the most relevant scenarios with guidance and "
            "similarity scores. No LLM call: fast and free."
        ),
    )
    def search_troubleshooting_kb(query: Query, top_k: TopK = 3) -> SearchResult:
        results = get_retriever().retrieve(query.strip(), top_k=top_k)
        in_scope = bool(results) and results[0]["score"] >= OUT_OF_SCOPE_THRESHOLD
        logger.info(
            "mcp search (top_k=%d, in_scope=%s)",
            top_k,
            in_scope,
            extra={"event": "mcp_search", "top_k": top_k},
        )
        if not in_scope:
            return SearchResult(query=query, in_scope=False, matches=[], note=OUT_OF_SCOPE_MESSAGE)
        return SearchResult(query=query, in_scope=True, matches=[_to_match(r) for r in results])

    @server.tool(
        name="ask_metricmedic",
        description=(
            "Get MetricMedic's full grounded answer (likely issue, recommended checks, why it "
            "happens) for a data troubleshooting question. Makes one LLM call."
        ),
    )
    async def ask_metricmedic(question: Query, top_k: TopK = 3) -> AskResult:
        # Off-topic questions are declined without spending an LLM call.
        top = get_retriever().retrieve(question.strip(), top_k=1)
        if not top or top[0]["score"] < OUT_OF_SCOPE_THRESHOLD:
            return AskResult(
                answer=OUT_OF_SCOPE_MESSAGE, sources=[], used_fallback=False, in_scope=False
            )

        def run() -> dict:
            try:
                return get_rag().answer(question, top_k=top_k, tags=["mcp"])
            finally:
                flush_traces()

        # The chain blocks for seconds; run it in a worker thread so the server stays responsive.
        result = await anyio.to_thread.run_sync(run)
        return AskResult(
            answer=result["answer"],
            sources=[_to_match(s) for s in result["sources"]],
            used_fallback=result["used_fallback"],
            in_scope=True,
        )

    @server.resource(
        "metricmedic://issues",
        name="knowledge_base_issues",
        description="Titles of every troubleshooting scenario in the knowledge base.",
        mime_type="application/json",
    )
    def knowledge_base_issues() -> str:
        return json.dumps([item["issue"] for item in get_retriever().knowledge_base])

    return server


server = build_server()


def main() -> None:
    parser = argparse.ArgumentParser(description="MetricMedic MCP server")
    parser.add_argument("--transport", choices=["stdio", "streamable-http"], default="stdio")
    args = parser.parse_args()
    setup_logging()  # logs to stderr, never stdout
    server.run(transport=args.transport)


if __name__ == "__main__":
    main()
