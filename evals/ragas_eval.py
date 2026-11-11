"""Step 2 of the LLM eval: score saved answers with RAGAS, using a LOCAL judge.

    python -m evals.ragas_eval --limit 2      # smoke test on 2 questions
    python -m evals.ragas_eval               # score everything not yet scored (resumable)
    python -m evals.ragas_eval --report-only # rebuild the report from saved scores

Judge LLM:   Ollama (default qwen2.5:7b) - free, local, no rate limits, and a different
             model from the one that wrote the answers (avoids self-grading bias).
Embeddings:  the same local MiniLM model the retriever uses (for answer relevancy).

Metrics (0 to 1, higher is better):
  faithfulness       are the answer's claims supported by the retrieved context?
  answer_relevancy   does the answer address the question that was asked?
  context_precision  are the relevant retrieved chunks ranked at the top?
  context_recall     does the retrieved context contain what the reference answer needs?

Out-of-scope questions are excluded: they have no meaningful reference context.
"""

import argparse
import asyncio
import math
import os
import statistics
import time
from datetime import datetime, timezone

# Opt out of RAGAS usage analytics: an offline eval should not phone home.
os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")

from langchain_core.embeddings import Embeddings
from langchain_ollama import ChatOllama
from ragas.dataset_schema import SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ResponseRelevancy,
)
from ragas.run_config import RunConfig
from sentence_transformers import SentenceTransformer

from evals.common import RESULTS_DIR, append_jsonl, read_jsonl, write_json
from metricmedic.config import EMBEDDING_MODEL

ANSWERS_PATH = RESULTS_DIR / "answers.jsonl"
SCORES_PATH = RESULTS_DIR / "ragas_scores.jsonl"
METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

JUDGE_MODEL = os.getenv("RAGAS_JUDGE_MODEL", "qwen2.5:7b")


class LocalEmbeddings(Embeddings):
    """LangChain Embeddings backed by the local Sentence Transformers model."""

    def __init__(self, model_name: str):
        # RAGAS reads `.model` as the model NAME (a string) for its usage events,
        # so the loaded model object lives under a different attribute.
        self.model = model_name
        self._encoder = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encoder.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def build_metrics(timeout: int) -> dict:
    judge = LangchainLLMWrapper(
        ChatOllama(
            model=JUDGE_MODEL,
            temperature=0,
            num_ctx=8192,  # Ollama's default 2048 tokens would truncate long prompts
            format="json",  # RAGAS expects JSON replies; this forces valid JSON
        )
    )
    embeddings = LangchainEmbeddingsWrapper(LocalEmbeddings(EMBEDDING_MODEL))
    run_config = RunConfig(timeout=timeout, max_retries=2, max_workers=1)

    metrics = {
        "faithfulness": Faithfulness(llm=judge),
        # strictness=1: one generated question instead of 3 -> fewer judge calls
        "answer_relevancy": ResponseRelevancy(llm=judge, embeddings=embeddings, strictness=1),
        "context_precision": LLMContextPrecisionWithReference(llm=judge),
        "context_recall": LLMContextRecall(llm=judge),
    }
    for metric in metrics.values():
        metric.init(run_config)
    return metrics


def latest_answers() -> list[dict]:
    """Newest non-fallback answer per in-scope question."""
    latest = {}
    for row in read_jsonl(ANSWERS_PATH):
        if row["used_fallback"] or row["category"] == "out_of_scope":
            continue
        latest[row["id"]] = row
    return sorted(latest.values(), key=lambda r: r["id"])


def clean(value):
    if value is None:
        return None
    value = float(value)
    return None if math.isnan(value) else round(value, 4)


async def score_one(row: dict, metrics: dict, timeout: int) -> dict:
    sample = SingleTurnSample(
        user_input=row["question"],
        response=row["answer"],
        retrieved_contexts=row["retrieved_contexts"],
        reference=row["reference"],
    )
    scores, errors = {}, {}
    for name, metric in metrics.items():
        try:
            value = await asyncio.wait_for(metric.single_turn_ascore(sample), timeout=timeout)
            scores[name] = clean(value)
            if scores[name] is None:
                errors[name] = "judge returned no score (NaN)"
        except Exception as exc:  # one bad metric must not stop the run
            scores[name] = None
            errors[name] = f"{type(exc).__name__}: {str(exc)[:200]}"
    return {"scores": scores, "errors": errors}


def summarize(scored: list[dict]) -> dict:
    def mean_of(rows, name):
        values = [r["scores"][name] for r in rows if r["scores"].get(name) is not None]
        return (round(statistics.mean(values), 3), len(values)) if values else (None, 0)

    overall = {}
    for name in METRIC_NAMES:
        value, n = mean_of(scored, name)
        overall[name] = {"mean": value, "scored": n, "of": len(scored)}

    by_category = {}
    for category in sorted({r["category"] for r in scored}):
        rows = [r for r in scored if r["category"] == category]
        by_category[category] = {name: mean_of(rows, name)[0] for name in METRIC_NAMES}
        by_category[category]["n"] = len(rows)

    lowest = {}
    for name in METRIC_NAMES:
        rows = [r for r in scored if r["scores"].get(name) is not None]
        rows.sort(key=lambda r: r["scores"][name])
        lowest[name] = [
            {"id": r["id"], "score": r["scores"][name], "question": r["question"]} for r in rows[:3]
        ]
    return {"overall": overall, "by_category": by_category, "lowest": lowest}


def to_markdown(report: dict) -> str:
    s = report["summary"]
    lines = [
        "# RAGAS evaluation",
        "",
        f"Run: {report['run_at']} · answers by `{report['answer_model']}` · judged by "
        f"`{report['judge_model']}` (local, Ollama) · {report['questions']} in-scope questions",
        "",
        "| Metric | Mean | Scored |",
        "| --- | --- | --- |",
    ]
    for name in METRIC_NAMES:
        m = s["overall"][name]
        mean = "n/a" if m["mean"] is None else f"{m['mean']:.3f}"
        lines.append(f"| {name} | {mean} | {m['scored']}/{m['of']} |")

    lines += ["", "## By category", "", "| Category | n | " + " | ".join(METRIC_NAMES) + " |"]
    lines.append("| --- | --- | " + " | ".join("---" for _ in METRIC_NAMES) + " |")
    for category, values in s["by_category"].items():
        cells = ["n/a" if values[n] is None else f"{values[n]:.2f}" for n in METRIC_NAMES]
        lines.append(f"| {category} | {values['n']} | " + " | ".join(cells) + " |")

    lines += ["", "## Lowest-scoring questions", ""]
    for name in METRIC_NAMES:
        items = ", ".join(f"{x['id']} ({x['score']:.2f})" for x in s["lowest"][name])
        lines.append(f"- **{name}:** {items or 'n/a'}")
    return "\n".join(lines) + "\n"


def write_report() -> dict:
    scored = read_jsonl(SCORES_PATH)
    report = {
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "judge_model": JUDGE_MODEL,
        "answer_model": scored[0]["answer_model"] if scored else None,
        "questions": len(scored),
        "summary": summarize(scored) if scored else {},
        "per_question": scored,
    }
    write_json(RESULTS_DIR / "ragas_report.json", report)
    if scored:
        (RESULTS_DIR / "ragas_report.md").write_text(to_markdown(report), encoding="utf-8")
        print(to_markdown(report))
    return report


async def run(limit: int | None, timeout: int) -> None:
    answers = latest_answers()
    done = {r["id"] for r in read_jsonl(SCORES_PATH)}
    todo = [a for a in answers if a["id"] not in done]
    if limit:
        todo = todo[:limit]
    print(f"{len(done)} already scored, scoring {len(todo)} now (judge: {JUDGE_MODEL})")

    metrics = build_metrics(timeout)
    for n, row in enumerate(todo, start=1):
        start = time.perf_counter()
        result = await score_one(row, metrics, timeout)
        seconds = round(time.perf_counter() - start, 1)
        append_jsonl(
            SCORES_PATH,
            {
                "id": row["id"],
                "category": row["category"],
                "question": row["question"],
                "answer_model": row["model"],
                "judge_model": JUDGE_MODEL,
                "scores": result["scores"],
                "errors": result["errors"],
                "seconds": seconds,
            },
        )
        shown = " ".join(
            f"{k[:5]}={'-' if v is None else f'{v:.2f}'}" for k, v in result["scores"].items()
        )
        warn = f"  ({len(result['errors'])} errors)" if result["errors"] else ""
        print(f"[{n}/{len(todo)}] {row['id']} {shown}  {seconds}s{warn}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="score at most N questions")
    parser.add_argument("--timeout", type=int, default=300, help="seconds per metric call")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()

    if not args.report_only:
        asyncio.run(run(args.limit, args.timeout))
    write_report()
    print(f"Saved to {RESULTS_DIR}/ragas_report.(json|md)")


if __name__ == "__main__":
    main()
