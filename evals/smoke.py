"""Eval smoke tests for CI. Exit code 1 = quality regression.

    python -m evals.smoke --retrieval       # free: no LLM, deterministic quality gate
    python -m evals.smoke --llm 3           # 3 real questions through Gemini (needs key)

The retrieval gate fails the build if retrieval quality drops below the
Milestone 4 baseline (minus a small margin), or if the out-of-scope threshold
used by the MCP server no longer separates on-topic from off-topic questions.
"""

import argparse
import sys

from evals.common import load_golden
from evals.retrieval_eval import evaluate
from metricmedic import MetricMedicRAG, MetricMedicRetriever
from metricmedic.mcp_server import OUT_OF_SCOPE_THRESHOLD

# Baseline (Milestone 4): hit@3 0.838, MRR 0.719. Gates leave a small margin.
MIN_HIT_AT_3 = 0.80
MIN_MRR = 0.65


def retrieval_gate() -> list[str]:
    report = evaluate(MetricMedicRetriever(), load_golden())
    s = report["summary"]
    print(
        f"hit@3={s['hit@3']:.3f} (min {MIN_HIT_AT_3})  mrr={s['mrr']:.3f} (min {MIN_MRR})  "
        f"in-scope top-1 min={s['in_scope_top1_score_min']:.3f}  "
        f"out-of-scope top-1 max={s.get('out_of_scope_top1_score_max', 0):.3f}  "
        f"threshold={OUT_OF_SCOPE_THRESHOLD}"
    )
    failures = []
    if s["hit@3"] < MIN_HIT_AT_3:
        failures.append(f"hit@3 {s['hit@3']:.3f} < {MIN_HIT_AT_3}")
    if s["mrr"] < MIN_MRR:
        failures.append(f"MRR {s['mrr']:.3f} < {MIN_MRR}")
    if s["in_scope_top1_score_min"] < OUT_OF_SCOPE_THRESHOLD:
        failures.append("an in-scope question falls below the out-of-scope threshold")
    if s.get("out_of_scope_top1_score_max", 0) >= OUT_OF_SCOPE_THRESHOLD:
        failures.append("an out-of-scope question scores above the out-of-scope threshold")
    return failures


def llm_smoke(n: int) -> list[str]:
    questions = [g for g in load_golden() if g["category"] == "single"][:n]
    rag = MetricMedicRAG()
    failures = []
    for item in questions:
        result = rag.answer(item["question"], tags=["ci-smoke"])
        ok = (
            not result["used_fallback"]
            and "Likely issue" in result["answer"]
            and len(result["sources"]) == 3
        )
        print(f"{item['id']}: {'ok' if ok else 'FAIL'} (fallback={result['used_fallback']})")
        if not ok:
            failures.append(f"{item['id']} did not produce a grounded Gemini answer")
    return failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval", action="store_true", help="run the retrieval gate")
    parser.add_argument("--llm", type=int, default=0, help="number of live Gemini questions")
    args = parser.parse_args()

    failures = []
    if args.retrieval:
        failures += retrieval_gate()
    if args.llm:
        failures += llm_smoke(args.llm)

    if failures:
        print("\nSMOKE TEST FAILED:\n- " + "\n- ".join(failures))
        sys.exit(1)
    print("\nSmoke test passed.")


if __name__ == "__main__":
    main()
