"""Step 1 of the LLM eval: run every golden question through the real pipeline.

    python -m evals.generate_answers            # resume: skips questions already answered
    python -m evals.generate_answers --fresh    # start over (deletes previous answers)

- Uses the production pipeline (retriever + chain + Gemini model from GEMINI_MODEL).
- Saves one JSON line per answer immediately, so a crash or quota stop loses nothing.
- Paces calls to stay under the free-tier requests-per-minute limit.
- Stops early if Gemini keeps failing (likely quota), instead of burning the run on fallbacks.
- Each answer is traced in Langfuse with tag "eval" and its golden ID in metadata.
"""

import argparse
import time
import uuid

from evals.common import RESULTS_DIR, append_jsonl, load_golden, read_jsonl
from metricmedic import MetricMedicRAG
from metricmedic.config import GEMINI_MODEL
from metricmedic.logging_config import setup_logging
from metricmedic.tracing import flush_traces

ANSWERS_PATH = RESULTS_DIR / "answers.jsonl"


def context_text(source: dict) -> str:
    """The retrieved scenario as the judge will see it."""
    return f"Issue: {source['issue']}\nGuidance: {source['response']}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sleep", type=float, default=6.0, help="seconds between Gemini calls")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--fresh", action="store_true", help="delete previous answers first")
    parser.add_argument("--max-fallbacks", type=int, default=2, help="stop after N in a row")
    args = parser.parse_args()

    setup_logging()
    if args.fresh and ANSWERS_PATH.exists():
        ANSWERS_PATH.unlink()

    golden = load_golden()
    done = {row["id"] for row in read_jsonl(ANSWERS_PATH) if not row["used_fallback"]}
    todo = [g for g in golden if g["id"] not in done]
    print(f"{len(done)} already answered, {len(todo)} to go (model: {GEMINI_MODEL})")
    if not todo:
        return

    rag = MetricMedicRAG()
    run_id = uuid.uuid4().hex[:8]
    consecutive_fallbacks = 0

    try:
        for n, item in enumerate(todo, start=1):
            start = time.perf_counter()
            result = rag.answer(
                item["question"],
                top_k=args.top_k,
                session_id=f"eval-{run_id}",
                tags=["eval"],
                trace_metadata={"golden_id": item["id"], "category": item["category"]},
            )
            latency_ms = round((time.perf_counter() - start) * 1000, 1)

            append_jsonl(
                ANSWERS_PATH,
                {
                    "id": item["id"],
                    "category": item["category"],
                    "question": item["question"],
                    "reference": item["reference"],
                    "relevant_issues": item["relevant_issues"],
                    "answer": result["answer"],
                    "retrieved_issues": [s["issue"] for s in result["sources"]],
                    "retrieved_contexts": [context_text(s) for s in result["sources"]],
                    "used_fallback": result["used_fallback"],
                    "latency_ms": latency_ms,
                    "model": GEMINI_MODEL,
                    "top_k": args.top_k,
                    "run_id": run_id,
                },
            )
            flag = "FALLBACK" if result["used_fallback"] else "ok"
            print(f"[{n}/{len(todo)}] {item['id']} {flag} {latency_ms:.0f} ms")

            consecutive_fallbacks = consecutive_fallbacks + 1 if result["used_fallback"] else 0
            if consecutive_fallbacks >= args.max_fallbacks:
                print(
                    f"\nStopped: {consecutive_fallbacks} fallbacks in a row - Gemini is failing "
                    "(often the daily quota). Check the WARNING lines above, then re-run later; "
                    "answered questions are kept and fallback answers will be retried."
                )
                break

            if n < len(todo):
                time.sleep(args.sleep)
    finally:
        flush_traces()

    rows = read_jsonl(ANSWERS_PATH)
    good = {r["id"] for r in rows if not r["used_fallback"]}
    print(f"\nDone: {len(good)}/{len(golden)} questions have a real Gemini answer.")
    print(f"Saved to {ANSWERS_PATH}")


if __name__ == "__main__":
    main()
