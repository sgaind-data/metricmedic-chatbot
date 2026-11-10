"""The RAG chain, built with LCEL.

question
  -> {question, docs}             (retriever)
  -> + sources, context           (formatting)
  -> + generation                 (prompt | Gemini | parser, with logged fallback)
  -> {answer, sources, used_fallback}
"""

import logging

from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import (
    Runnable,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)

from metricmedic.lc_retriever import documents_to_sources
from metricmedic.prompts import (
    NO_CONTEXT_ANSWER,
    RAG_PROMPT,
    fallback_answer,
    format_context,
)

logger = logging.getLogger("metricmedic.chain")


def _fallback(inputs: dict) -> dict:
    """Runs when Gemini fails. Logs the real error so it is never silent."""
    error = inputs.get("error")
    logger.warning(
        "LLM generation failed; using retrieved-context fallback: %s: %s",
        type(error).__name__,
        str(error)[:300],
        extra={"event": "llm_fallback", "error_type": type(error).__name__},
    )
    return {"text": fallback_answer(inputs["sources"]), "used_fallback": True}


def build_rag_chain(retriever: BaseRetriever, llm: Runnable) -> Runnable:
    """Input: question str. Output: {answer, sources, used_fallback}."""

    generate = (
        RAG_PROMPT
        | llm
        | StrOutputParser()
        | RunnableLambda(lambda text: {"text": text, "used_fallback": False})
    ).with_config(run_name="generate_answer")

    # exception_key="error" passes the caught exception into _fallback's input.
    safe_generate = generate.with_fallbacks(
        [RunnableLambda(_fallback, name="fallback_answer")],
        exception_key="error",
    )

    def route(inputs: dict):
        if not inputs["sources"]:
            return {"text": NO_CONTEXT_ANSWER, "used_fallback": False}
        return safe_generate

    chain = (
        RunnableParallel(question=RunnablePassthrough(), docs=retriever)
        | RunnablePassthrough.assign(sources=lambda x: documents_to_sources(x["docs"]))
        | RunnablePassthrough.assign(context=lambda x: format_context(x["sources"]))
        | RunnablePassthrough.assign(generation=RunnableLambda(route, name="route"))
        | RunnableLambda(
            lambda x: {
                "answer": x["generation"]["text"],
                "sources": x["sources"],
                "used_fallback": x["generation"]["used_fallback"],
            },
            name="format_output",
        )
    )
    return chain.with_config(run_name="metricmedic_rag")
