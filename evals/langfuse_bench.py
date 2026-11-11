"""Latency and token benchmark, measured from Langfuse traces (not local timers).

    python -m evals.langfuse_bench run --n 60           # 60 real Gemini calls, tag bench-<time>
    python -m evals.langfuse_bench stats --tag bench-.. # p50 / p95 / tokens / cost from Langfuse

`run` sends in-scope golden questions through the full pipeline (cycling if n > 37).
`stats` reads the traces back from the Langfuse public API, so every number it prints
can be checked in the Langfuse UI by filtering on the same tag.
Uses the v2 Observations API (the v1 /traces endpoint now returns 410 Gone).
"""

import argparse
import json
import math
import os
import statistics
import sys
import time
from datetime import UTC, datetime, timedelta

import httpx
from dotenv import load_dotenv

from evals.common import RESULTS_DIR, load_golden, write_json


def run(n: int, sleep_s: float) -> None:
    from metricmedic import MetricMedicRAG
    from metricmedic.tracing import flush_traces, tracing_enabled

    if not tracing_enabled():
        sys.exit("Langfuse keys are missing from .env, so nothing would be traced.")

    questions = [g["question"] for g in load_golden() if g["category"] != "out_of_scope"]
    run_tag = time.strftime("bench-%Y%m%d-%H%M")
    rag = MetricMedicRAG()
    fallbacks = consecutive = 0
    try:
        for i in range(n):
            result = rag.answer(questions[i % len(questions)], tags=["bench", run_tag])
            if result["used_fallback"]:
                fallbacks += 1
                consecutive += 1
            else:
                consecutive = 0
            print(f"{i + 1}/{n} fallback={result['used_fallback']}")
            if consecutive >= 2:
                print("Two fallbacks in a row (likely rate limit). Stopping early.")
                break
            time.sleep(sleep_s)
    finally:
        flush_traces()

    print(f"\nTag: {run_tag}   fallbacks: {fallbacks}")
    if fallbacks:
        print("WARNING: fallback answers skip Gemini and are fast; rerun later for clean latency.")
    print(f"Wait ~2 minutes, then: python -m evals.langfuse_bench stats --tag {run_tag}")


def _get_all(client: httpx.Client, params: dict) -> list[dict]:
    """GET /api/public/v2/observations, following meta.cursor until it is absent."""
    rows, params = [], dict(params)
    while True:
        for _ in range(5):
            response = client.get("/api/public/v2/observations", params=params)
            if response.status_code != 429:
                break
            time.sleep(5)
        response.raise_for_status()
        body = response.json()
        rows += body.get("data", [])
        cursor = (body.get("meta") or {}).get("cursor")
        if not cursor:
            return rows
        params["cursor"] = cursor


def _ms(start: str, end: str) -> float:
    return (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds() * 1000


def _tokens(row: dict) -> tuple[int, int]:
    details = row.get("usageDetails") or {}
    tokens_in = row.get("inputUsage") or details.get("input") or 0
    tokens_out = row.get("outputUsage") or details.get("output") or 0
    return int(tokens_in), int(tokens_out)


def stats(tag: str, hours: int) -> None:
    load_dotenv()
    base_url = os.getenv("LANGFUSE_BASE_URL") or os.getenv("LANGFUSE_HOST")
    if not base_url:
        sys.exit("LANGFUSE_BASE_URL is not set in .env.")
    auth = (os.environ["LANGFUSE_PUBLIC_KEY"], os.environ["LANGFUSE_SECRET_KEY"])
    now = datetime.now(UTC)
    window = {
        "fromStartTime": (now - timedelta(hours=hours)).isoformat().replace("+00:00", "Z"),
        "toStartTime": (now + timedelta(minutes=5)).isoformat().replace("+00:00", "Z"),
        "filter": json.dumps(
            [{"type": "arrayOptions", "column": "tags", "operator": "any of", "value": [tag]}]
        ),
        "limit": 1000,
    }

    with httpx.Client(base_url=base_url.rstrip("/"), auth=auth, timeout=60) as client:
        # One root observation per trace = the whole request, start to end.
        roots = _get_all(client, {**window, "isRootObservation": "true", "fields": "core,basic"})
        # Every Gemini call, with token usage and Langfuse-computed cost.
        generations = _get_all(client, {**window, "type": "GENERATION", "fields": "core,usage"})

    latencies_ms = sorted(_ms(r["startTime"], r["endTime"]) for r in roots if r.get("endTime"))
    n = len(latencies_ms)
    if n == 0:
        sys.exit(f"No finished traces found for tag {tag!r}. Wait a minute and retry.")
    tokens_in = sum(_tokens(g)[0] for g in generations)
    tokens_out = sum(_tokens(g)[1] for g in generations)
    cost = sum(float(g.get("totalCost") or 0) for g in generations)

    summary = {
        "tag": tag,
        "traces": n,
        "gemini_calls": len(generations),
        "p50_ms": round(statistics.median(latencies_ms)),
        "p95_ms": round(latencies_ms[math.ceil(0.95 * n) - 1]),  # nearest-rank p95
        "max_ms": round(latencies_ms[-1]),
        "tokens_in_per_100": round(tokens_in / n * 100),
        "tokens_out_per_100": round(tokens_out / n * 100),
        "tokens_total_per_100": round((tokens_in + tokens_out) / n * 100),
        "langfuse_cost_usd_per_100": round(cost / n * 100, 4),
    }
    for key, value in summary.items():
        print(f"{key:28} {value}")
    if n < 50:
        print("\nWARNING: fewer than 50 traces; run more before quoting percentiles.")
    write_json(RESULTS_DIR / f"langfuse_bench_{tag}.json", summary)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--n", type=int, default=60)
    run_parser.add_argument("--sleep", type=float, default=5.0, help="seconds between calls")
    stats_parser = sub.add_parser("stats")
    stats_parser.add_argument("--tag", required=True)
    stats_parser.add_argument("--hours", type=int, default=24, help="look-back window")
    args = parser.parse_args()
    if args.command == "run":
        run(args.n, args.sleep)
    else:
        stats(args.tag, args.hours)


if __name__ == "__main__":
    main()
