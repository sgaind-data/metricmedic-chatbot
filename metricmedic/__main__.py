"""CLI smoke test:  python -m metricmedic "My dashboard count doubled after a join"
(Makes ONE Gemini call.)"""

import sys

from metricmedic import MetricMedicRAG
from metricmedic.logging_config import setup_logging
from metricmedic.tracing import flush_traces


def main():
    setup_logging()
    query = " ".join(sys.argv[1:]) or "My dashboard customer count doubled after joining two tables"
    try:
        result = MetricMedicRAG().answer(query, tags=["cli"])
    finally:
        # Short-lived script: send buffered traces before Python exits.
        flush_traces()
    print("\nAnswer:\n")
    print(result["answer"])
    print(f"\nUsed fallback: {result['used_fallback']}")
    print("\nRetrieved issues:")
    for s in result["sources"]:
        print(f"- {s['issue']} ({s['score']:.4f})")


if __name__ == "__main__":
    main()
