"""Retrieval evaluation: FAISS only, no LLM calls, deterministic, free.

    python -m evals.retrieval_eval

Metrics (in-scope questions only):
  hit@k     share of questions where at least one relevant issue is in the top k
  MRR       mean reciprocal rank of the first relevant issue (1.0 = always rank 1)
  recall@3  share of each question's relevant issues found in the top 3, averaged

Out-of-scope questions have no relevant issues; for them we report the top-1
similarity score, to see whether a score threshold could reject them.
"""

import statistics
from datetime import datetime, timezone

from evals.common import RESULTS_DIR, load_golden, write_json
from metricmedic import MetricMedicRetriever

KS = (1, 3, 5)


def evaluate(retriever: MetricMedicRetriever, golden: list[dict]) -> dict:
    max_k = max(KS)
    per_question = []

    for item in golden:
        results = retriever.retrieve(item["question"], top_k=max_k)
        ranked = [r["issue"] for r in results]
        relevant = set(item["relevant_issues"])
        first_rank = next((i + 1 for i, issue in enumerate(ranked) if issue in relevant), None)
        per_question.append(
            {
                "id": item["id"],
                "category": item["category"],
                "question": item["question"],
                "relevant_issues": item["relevant_issues"],
                "retrieved": [
                    {"issue": r["issue"], "score": round(r["score"], 4)} for r in results
                ],
                "first_relevant_rank": first_rank,
                "top1_score": round(results[0]["score"], 4) if results else None,
                "recall_at_3": (
                    len(relevant & set(ranked[:3])) / len(relevant) if relevant else None
                ),
            }
        )

    in_scope = [q for q in per_question if q["relevant_issues"]]
    out_scope = [q for q in per_question if not q["relevant_issues"]]

    def hit_at(k):
        hits = [q for q in in_scope if q["first_relevant_rank"] and q["first_relevant_rank"] <= k]
        return len(hits) / len(in_scope)

    summary = {f"hit@{k}": round(hit_at(k), 3) for k in KS}
    reciprocal_ranks = [
        1 / q["first_relevant_rank"] if q["first_relevant_rank"] else 0 for q in in_scope
    ]
    summary["mrr"] = round(statistics.mean(reciprocal_ranks), 3)
    summary["recall@3"] = round(statistics.mean(q["recall_at_3"] for q in in_scope), 3)
    summary["in_scope_questions"] = len(in_scope)
    summary["in_scope_top1_score_min"] = min(q["top1_score"] for q in in_scope)
    top1_scores = [q["top1_score"] for q in in_scope]
    summary["in_scope_top1_score_mean"] = round(statistics.mean(top1_scores), 4)
    if out_scope:
        summary["out_of_scope_questions"] = len(out_scope)
        summary["out_of_scope_top1_score_max"] = max(q["top1_score"] for q in out_scope)

    misses = [q for q in in_scope if not q["first_relevant_rank"] or q["first_relevant_rank"] > 3]
    return {"summary": summary, "misses_at_3": misses, "per_question": per_question}


def to_markdown(report: dict) -> str:
    s = report["summary"]
    lines = [
        "# Retrieval evaluation",
        "",
        f"Run: {report['run_at']} · embedding model `{report['embedding_model']}` · "
        f"{s['in_scope_questions']} in-scope questions · no LLM calls",
        "",
        "| Metric | Value |",
        "| --- | --- |",
    ]
    for key in ("hit@1", "hit@3", "hit@5", "mrr", "recall@3"):
        lines.append(f"| {key} | {s[key]:.3f} |")
    lines += [
        f"| top-1 score, in-scope (min / mean) | {s['in_scope_top1_score_min']:.3f} / "
        f"{s['in_scope_top1_score_mean']:.3f} |",
    ]
    if "out_of_scope_top1_score_max" in s:
        oos_max = s["out_of_scope_top1_score_max"]
        lines.append(f"| top-1 score, out-of-scope (max) | {oos_max:.3f} |")

    lines += ["", "## Misses (no relevant issue in the top 3)", ""]
    if not report["misses_at_3"]:
        lines.append("None.")
    for q in report["misses_at_3"]:
        got = ", ".join(f"{r['issue']} ({r['score']:.2f})" for r in q["retrieved"][:3])
        expected = ", ".join(q["relevant_issues"])
        lines.append(f"- **{q['id']}** {q['question']}  ")
        lines.append(f"  expected: {expected}  ")
        lines.append(f"  got: {got}")
    return "\n".join(lines) + "\n"


def main():
    retriever = MetricMedicRetriever()
    report = evaluate(retriever, load_golden())
    report["run_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    report["embedding_model"] = retriever.model_name

    write_json(RESULTS_DIR / "retrieval_report.json", report)
    (RESULTS_DIR / "retrieval_report.md").write_text(to_markdown(report), encoding="utf-8")

    print(to_markdown(report))
    print(f"Saved to {RESULTS_DIR}/retrieval_report.(json|md)")


if __name__ == "__main__":
    main()
