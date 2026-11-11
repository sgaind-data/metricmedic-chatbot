# Retrieval comparison: dense vs hybrid (dense + BM25, RRF)

Run: 2026-11-11T23:51:32+00:00 · 37 in-scope questions · no LLM calls

| Metric | Dense (before) | Hybrid (after) | Change |
| --- | --- | --- | --- |
| hit@1 | 0.595 | 0.568 | -0.027 |
| hit@3 | 0.838 | 0.865 | +0.027 |
| hit@5 | 0.892 | 0.892 | +0.000 |
| mrr | 0.719 | 0.714 | -0.005 |
| recall@3 | 0.550 | 0.606 | +0.056 |

- Fixed at rank ≤ 3: q17, q22
- Broken at rank ≤ 3: q11
- Remaining hybrid misses: q10, q11, q19, q27, q30
- 0.25 out-of-scope threshold still separates: True
- **Decision (pre-registered rule): keep dense**
