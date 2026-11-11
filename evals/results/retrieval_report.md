# Retrieval evaluation

Run: 2026-11-11T20:28:27+00:00 · embedding model `all-MiniLM-L6-v2` · 37 in-scope questions · no LLM calls

| Metric | Value |
| --- | --- |
| hit@1 | 0.595 |
| hit@3 | 0.838 |
| hit@5 | 0.892 |
| mrr | 0.719 |
| recall@3 | 0.550 |
| top-1 score, in-scope (min / mean) | 0.333 / 0.522 |
| top-1 score, out-of-scope (max) | 0.142 |

## Misses (no relevant issue in the top 3)

- **q10** Sales and marketing report different numbers for 'active customers'.
  expected: KPI definition mismatch, Source and dashboard use different status logic
  got: COUNT and COUNT DISTINCT mismatch (0.47), Aggregation grain mismatch (0.38), Power BI dashboard mismatch (0.31)
- **q17** Joining on account_id returns nothing; one table stores it as text and the other as a number.
  expected: Data type mismatch
  got: Whitespace causing join mismatch (0.47), Incorrect SQL join (0.42), Missing records caused by NULL join keys (0.41)
- **q19** 'ACME' and 'Acme' are being treated as two different customers.
  expected: Case sensitivity mismatch
  got: Aggregation grain mismatch (0.33), COUNT and COUNT DISTINCT mismatch (0.30), Currency conversion mismatch (0.28)
- **q22** Totals differ by a few cents between the dashboard and the finance export.
  expected: Rounding differences
  got: Revenue does not match finance report (0.58), Dashboard total is higher than the source system (0.57), Metric uses different source tables (0.51)
- **q27** After combining two regional tables, some orders appear twice.
  expected: Duplicate records after UNION
  got: Incorrect GROUP BY logic (0.47), Many-to-many join causing row explosion (0.43), Aggregation grain mismatch (0.43)
- **q30** After a customer moved regions, their old orders are counted twice.
  expected: Slowly changing dimension affects history, Duplicate dimension values
  got: COUNT and COUNT DISTINCT mismatch (0.41), Aggregation grain mismatch (0.36), Duplicate records (0.33)
