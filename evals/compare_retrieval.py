"""Before/after retrieval comparison: dense (baseline) vs hybrid (dense + BM25, RRF).

    python -m evals.compare_retrieval

Decision rule, written down BEFORE running (so the golden set is not used to tune):
  adopt hybrid only if hit@3 AND MRR both improve, no previously-correct question is
  lost at rank <= 3, and the 0.25 out-of-scope threshold still separates the questions.
  Otherwise keep dense and report the negative result.

Parameters are fixed library defaults (see metricmedic/hybrid.py); this runs once.
Writes evals/results/retrieval_comparison.{json,md}.
"""

from datetime import UTC, datetime

from evals.common import RESULTS_DIR, load_golden, write_json
from evals.retrieval_eval import evaluate
from metricmedic import MetricMedicRetriever
from metricmedic.hybrid import HybridRetriever
from metricmedic.mcp_server import OUT_OF_SCOPE_THRESHOLD

METRICS = ("hit@1", "hit@3", "hit@5", "mrr", "recall@3")


def _ranks(report: dict) -> dict[str, int | None]:
    return {
        q["id"]: q["first_relevant_rank"] for q in report["per_question"] if q["relevant_issues"]
    }


def _hit3(rank: int | None) -> bool:
    return rank is not None and rank <= 3


def main() -> None:
    golden = load_golden()
    dense = evaluate(MetricMedicRetriever(), golden)
    hybrid = evaluate(HybridRetriever(), golden)
    d, h = dense["summary"], hybrid["summary"]

    d_ranks, h_ranks = _ranks(dense), _ranks(hybrid)
    fixed = sorted(q for q in d_ranks if not _hit3(d_ranks[q]) and _hit3(h_ranks[q]))
    broken = sorted(q for q in d_ranks if _hit3(d_ranks[q]) and not _hit3(h_ranks[q]))
    separates = (
        h["in_scope_top1_score_min"] >= OUT_OF_SCOPE_THRESHOLD
        and h.get("out_of_scope_top1_score_max", 0) < OUT_OF_SCOPE_THRESHOLD
    )
    adopt = h["hit@3"] > d["hit@3"] and h["mrr"] > d["mrr"] and not broken and separates

    result = {
        "run": datetime.now(UTC).isoformat(timespec="seconds"),
        "questions": d["in_scope_questions"],
        "dense": {m: d[m] for m in METRICS},
        "hybrid": {m: h[m] for m in METRICS},
        "fixed_at_3": fixed,
        "broken_at_3": broken,
        "threshold_still_separates": separates,
        "decision": "adopt hybrid" if adopt else "keep dense",
        "hybrid_misses_at_3": [q["id"] for q in hybrid["misses_at_3"]],
    }
    write_json(RESULTS_DIR / "retrieval_comparison.json", result)

    lines = [
        "# Retrieval comparison: dense vs hybrid (dense + BM25, RRF)",
        "",
        f"Run: {result['run']} · {result['questions']} in-scope questions · no LLM calls",
        "",
        "| Metric | Dense (before) | Hybrid (after) | Change |",
        "| --- | --- | --- | --- |",
    ]
    for m in METRICS:
        lines.append(f"| {m} | {d[m]:.3f} | {h[m]:.3f} | {h[m] - d[m]:+.3f} |")
    lines += [
        "",
        f"- Fixed at rank ≤ 3: {', '.join(fixed) or 'none'}",
        f"- Broken at rank ≤ 3: {', '.join(broken) or 'none'}",
        f"- Remaining hybrid misses: {', '.join(result['hybrid_misses_at_3']) or 'none'}",
        f"- 0.25 out-of-scope threshold still separates: {separates}",
        f"- **Decision (pre-registered rule): {result['decision']}**",
        "",
    ]
    (RESULTS_DIR / "retrieval_comparison.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
