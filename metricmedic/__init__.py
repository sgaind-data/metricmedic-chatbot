"""MetricMedic: RAG for SQL / dashboard / KPI / ETL troubleshooting."""

from metricmedic.pipeline import InvalidQueryError, MetricMedicRAG
from metricmedic.retriever import MetricMedicRetriever

__all__ = ["InvalidQueryError", "MetricMedicRAG", "MetricMedicRetriever"]
__version__ = "0.5.0"
