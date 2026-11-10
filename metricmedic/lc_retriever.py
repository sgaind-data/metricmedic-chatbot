"""LangChain wrapper around the FAISS retriever.

Exposing it as a BaseRetriever means it plugs into LCEL chains, Langfuse's
LangChain callback (Milestone 3), RAGAS (Milestone 4), and agent tooling.
"""

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from metricmedic.config import DEFAULT_TOP_K
from metricmedic.retriever import MetricMedicRetriever


class MetricMedicLCRetriever(BaseRetriever):
    """LangChain retriever backed by MetricMedicRetriever (FAISS)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    core: MetricMedicRetriever
    top_k: int = DEFAULT_TOP_K

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        results = self.core.retrieve(query, top_k=self.top_k)
        return [
            Document(
                page_content=self.core.documents[r["kb_index"]],
                metadata=r,
            )
            for r in results
        ]


def documents_to_sources(docs: list[Document]) -> list[dict]:
    """Convert Documents back to the dicts the Streamlit UI already expects."""
    return [dict(doc.metadata) for doc in docs]
