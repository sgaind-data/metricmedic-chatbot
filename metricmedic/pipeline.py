"""High-level entry point used by Streamlit, the CLI, FastAPI and (later) MCP."""

import logging
import time

from langchain_core.runnables import Runnable

from metricmedic.chain import build_rag_chain
from metricmedic.config import DEFAULT_TOP_K
from metricmedic.lc_retriever import MetricMedicLCRetriever
from metricmedic.retriever import MetricMedicRetriever
from metricmedic.tracing import build_trace_config, merge_configs

logger = logging.getLogger("metricmedic.pipeline")

MAX_QUERY_CHARS = 2000
MIN_TOP_K, MAX_TOP_K = 1, 10


class InvalidQueryError(ValueError):
    """Raised for empty/oversized questions or out-of-range top_k.
    FastAPI turns this into an HTTP 422."""


def validate_request(query: str, top_k: int) -> str:
    if not isinstance(query, str) or not query.strip():
        raise InvalidQueryError("Question must be a non-empty string.")
    query = query.strip()
    if len(query) > MAX_QUERY_CHARS:
        raise InvalidQueryError(f"Question must be at most {MAX_QUERY_CHARS} characters.")
    if not isinstance(top_k, int) or not (MIN_TOP_K <= top_k <= MAX_TOP_K):
        raise InvalidQueryError(f"top_k must be an integer between {MIN_TOP_K} and {MAX_TOP_K}.")
    return query


class MetricMedicRAG:
    """Retrieval + grounded generation."""

    def __init__(self, retriever: MetricMedicRetriever | None = None, llm: Runnable | None = None):
        self.retriever = retriever or MetricMedicRetriever()
        if llm is None:
            from metricmedic.llm import get_chat_model

            llm = get_chat_model()
        self.llm = llm
        self._chains: dict[int, Runnable] = {}
        logger.info(
            "MetricMedicRAG ready (%d knowledge-base entries)",
            len(self.retriever.knowledge_base),
            extra={"event": "startup"},
        )

    def get_chain(self, top_k: int = DEFAULT_TOP_K) -> Runnable:
        """One chain per top_k, all sharing the same FAISS index and model."""
        if top_k not in self._chains:
            lc_retriever = MetricMedicLCRetriever(core=self.retriever, top_k=top_k)
            self._chains[top_k] = build_rag_chain(lc_retriever, self.llm)
        return self._chains[top_k]

    def answer(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        config: dict | None = None,
        *,
        session_id: str | None = None,
        user_id: str | None = None,
        tags: list[str] | None = None,
        trace_metadata: dict | None = None,
    ) -> dict:
        """Return {"answer": str, "sources": list[dict], "used_fallback": bool}.

        session_id / user_id / tags / trace_metadata are attached to the Langfuse
        trace when tracing is enabled, and ignored otherwise.
        """
        query = validate_request(query, top_k)

        run_config = build_trace_config(
            session_id=session_id,
            user_id=user_id,
            tags=tags,
            metadata={"top_k": top_k, **(trace_metadata or {})},
        )
        run_config = merge_configs(run_config, config)

        start = time.perf_counter()
        result = self.get_chain(top_k).invoke(query, config=run_config or None)
        latency_ms = round((time.perf_counter() - start) * 1000, 1)

        # Log sizes and timings, never the question text itself (it may contain
        # sensitive business data).
        logger.info(
            "answered in %.0f ms (top_k=%d, sources=%d, fallback=%s)",
            latency_ms,
            top_k,
            len(result["sources"]),
            result["used_fallback"],
            extra={
                "event": "rag_answer",
                "latency_ms": latency_ms,
                "top_k": top_k,
                "num_sources": len(result["sources"]),
                "used_fallback": result["used_fallback"],
                "query_chars": len(query),
            },
        )
        return result
