"""One-off helper used to author evals/golden_dataset.jsonl (kept for transparency).

Each item:
  id               stable ID
  category         single | multi | ambiguous | out_of_scope
  question         phrased like a real user (paraphrased, not copied from keywords)
  reference        ground-truth answer written from the knowledge base
  relevant_issues  KB issue titles that a good retriever should return (any = hit)
"""

import json
from pathlib import Path

ITEMS = [
    # --- single-issue, paraphrased -------------------------------------------------
    (
        "single",
        "Our Tableau dashboard shows more sales than the ERP system for the same week. What should I look at?",
        "Check for duplicate rows and one-to-many or many-to-many joins that multiply records, then compare filters, date ranges and aggregation logic between the dashboard and the source system.",
        [
            "Dashboard total is higher than the source system",
            "Duplicate records",
            "Incorrect SQL join",
        ],
    ),
    (
        "single",
        "The same customer appears three times in my customer extract.",
        "Identify the expected unique key and check whether it appears more than once. Review joins, because one-to-many relationships can multiply records, and confirm the dimension key is unique.",
        ["Duplicate records", "Duplicate dimension values"],
    ),
    (
        "single",
        "After I joined orders to order_items my row count went way up. Is my join wrong?",
        "Validate the join keys and the relationship between the tables and compare row counts before and after the join. If both tables have multiple rows per key, aggregate or deduplicate one side before joining.",
        ["Incorrect SQL join", "Many-to-many join causing row explosion", "Duplicate records"],
    ),
    (
        "single",
        "My average discount looks too high and I think blank values are involved.",
        "Check how NULL values are handled in the calculation and filters and confirm what NULL represents before replacing it with a default. Also check whether the metric needs a simple or weighted average and whether duplicates affect the denominator.",
        ["NULL values affecting calculations", "Incorrect average calculation"],
    ),
    (
        "single",
        "COUNT(*) gives 12,400 customers but COUNT(DISTINCT customer_id) gives 9,800. Which one is right?",
        "Compare COUNT(*) with COUNT(DISTINCT identifier). The difference may indicate duplicate rows or one-to-many joins; the distinct count of the identifier reflects unique customers.",
        ["COUNT and COUNT DISTINCT mismatch"],
    ),
    (
        "single",
        "Two reports for 'last month' show different order counts.",
        "Confirm both reports use the same date field, date range, timezone and inclusion rules, and whether one uses a fiscal calendar while the other uses calendar months.",
        [
            "Different date filters",
            "Timezone mismatch",
            "Fiscal calendar mismatch",
            "Date boundary off by one day",
        ],
    ),
    (
        "single",
        "My dashboard is still showing yesterday's numbers this afternoon.",
        "Check the last refresh timestamp and verify whether the ETL pipeline or scheduled refresh completed successfully. Also check whether the dashboard uses a cached extract and its refresh schedule.",
        ["Dashboard has stale data", "Dashboard uses cached extract", "Late-arriving data"],
    ),
    (
        "single",
        "Finance's revenue for Q3 is lower than what my dashboard reports.",
        "Confirm both teams use the same revenue definition and check refunds, cancelled transactions, discounts, pending payments and recognized revenue, as well as whether both use the same fiscal calendar.",
        [
            "Revenue does not match finance report",
            "Cancelled orders included in revenue",
            "Fiscal calendar mismatch",
        ],
    ),
    (
        "single",
        "The total row in my Power BI matrix doesn't equal the sum of the rows above it.",
        "Review the DAX measure's filter context and aggregation context, including any CALCULATE, ALL or FILTER logic, and compare the underlying rows with the source SQL.",
        ["Power BI dashboard mismatch", "DAX filter context issue"],
    ),
    (
        "single",
        "Sales and marketing report different numbers for 'active customers'.",
        "Document the KPI formula, population, exclusions, date logic and aggregation method and confirm the definition with the business owner. Compare the status rules each team uses for 'active'.",
        ["KPI definition mismatch", "Source and dashboard use different status logic"],
    ),
    (
        "single",
        "Some orders vanish from my report after joining to the customers table.",
        "Check whether unmatched records are removed by an INNER JOIN: test with a LEFT JOIN and inspect unmatched keys, check join columns for NULL values, and look for orphan records whose foreign keys have no matching dimension row.",
        [
            "Inner join dropping records",
            "Missing records caused by NULL join keys",
            "Missing foreign key relationships",
        ],
    ),
    (
        "single",
        "Joining two tables on product_id turned 1,000 rows into 50,000.",
        "Both tables likely contain multiple rows for the join key. Aggregate or deduplicate one side before joining if a many-to-many relationship is not intended, and validate row counts before and after the join.",
        ["Many-to-many join causing row explosion", "Incorrect SQL join"],
    ),
    (
        "single",
        "Yesterday's revenue on the dashboard is 40% lower than usual.",
        "Compare refresh times, source row counts, filters and ETL job status. A sudden drop may indicate missing source data, an incomplete pipeline load or late-arriving records.",
        [
            "Unexpected drop in dashboard values",
            "ETL pipeline loaded incomplete data",
            "Late-arriving data",
        ],
    ),
    (
        "single",
        "Hourly orders in my report are shifted by several hours compared with the app database.",
        "One system likely uses UTC while the other uses local time. Convert timestamps consistently before comparing daily or hourly metrics.",
        ["Timezone mismatch"],
    ),
    (
        "single",
        "My monthly customer-level report total doesn't match the transaction-level total.",
        "Confirm both reports aggregate at the same level of detail; customer-level, transaction-level and monthly data can produce different totals. Also check whether one is a point-in-time snapshot.",
        ["Aggregation grain mismatch", "Snapshot versus transactional data mismatch"],
    ),
    (
        "single",
        "Our average order value seems off when I average the regional averages.",
        "Check whether the metric requires a simple or a weighted average; averaging regional averages ignores their weights. Also verify whether duplicate rows are influencing the denominator.",
        ["Incorrect average calculation"],
    ),
    (
        "single",
        "Joining on account_id returns nothing; one table stores it as text and the other as a number.",
        "The joined fields use incompatible data types. Cast values consistently before joining or filtering.",
        ["Data type mismatch"],
    ),
    (
        "single",
        "Customer codes look identical but the join still doesn't match them.",
        "Check text fields for leading or trailing whitespace and trim them before joining. Also check case differences and whether both fields use the same data type.",
        ["Whitespace causing join mismatch", "Case sensitivity mismatch", "Data type mismatch"],
    ),
    (
        "single",
        "'ACME' and 'Acme' are being treated as two different customers.",
        "Normalize text values with UPPER() or LOWER() before comparison if the systems handle case differently.",
        ["Case sensitivity mismatch"],
    ),
    (
        "single",
        "Some transaction IDs appear twice after last night's load, doubling revenue for those orders.",
        "Check whether transaction IDs are unique and whether records were reloaded or appended more than once during ingestion.",
        ["Duplicate source transactions", "Duplicate records"],
    ),
    (
        "single",
        "USD revenue differs between our EU report and the global report.",
        "Verify both reports use the same exchange rate, conversion date, source currency and rounding rules.",
        ["Currency conversion mismatch", "Rounding differences"],
    ),
    (
        "single",
        "Totals differ by a few cents between the dashboard and the finance export.",
        "Check whether values are rounded before or after aggregation; different decimal precision can create small reporting differences.",
        ["Rounding differences"],
    ),
    (
        "single",
        "The visual shows fewer rows than my SQL query, but I can't see any filters on the page.",
        "Review report-level, page-level and visual-level filters; a hidden filter may exclude records present in the source query. Also check whether row-level security is filtering the data.",
        ["Hidden dashboard filter", "Row-level security affecting totals"],
    ),
    (
        "single",
        "My manager sees a bigger total than I do on the same Power BI report.",
        "Check whether row-level security filters data differently for different users or roles.",
        ["Row-level security affecting totals"],
    ),
    (
        "single",
        "Our nightly incremental load keeps missing orders created just before midnight.",
        "Verify the incremental-load watermark or timestamp condition; late-arriving or delayed records may be excluded by incorrect cutoff logic. Check inclusive versus exclusive boundaries around midnight.",
        ["Incremental load missed records", "Late-arriving data", "Date boundary off by one day"],
    ),
    (
        "single",
        "Last month's numbers changed even though nobody touched the report.",
        "Check whether source data is updated retrospectively, business logic changed, late-arriving data was backfilled, or slowly changing dimensions altered historical results.",
        [
            "Historical data changed unexpectedly",
            "Slowly changing dimension affects history",
            "Late-arriving data",
        ],
    ),
    (
        "single",
        "After combining two regional tables, some orders appear twice.",
        "Check whether UNION ALL is preserving duplicate rows when UNION is expected, or whether the source datasets already contain overlapping records.",
        ["Duplicate records after UNION"],
    ),
    (
        "single",
        "Adding one column to my query split a single total into many smaller rows.",
        "Review the GROUP BY columns and make sure they match the intended reporting grain; extra grouping columns split totals.",
        ["Incorrect GROUP BY logic", "Aggregation grain mismatch"],
    ),
    (
        "single",
        "Some product categories disappear from my aggregated results.",
        "Check whether the HAVING clause filters out aggregated groups that should remain, and test WHERE conditions independently to see which removes expected rows.",
        ["HAVING filter removes expected groups", "WHERE clause excludes valid rows"],
    ),
    (
        "single",
        "After a customer moved regions, their old orders are counted twice.",
        "Multiple dimension versions are likely joining to the same fact record. Apply effective-date logic for Type 2 slowly changing dimensions and verify the dimension key is unique.",
        ["Slowly changing dimension affects history", "Duplicate dimension values"],
    ),
    (
        "single",
        "My ROW_NUMBER query to get each customer's latest order returns several rows per customer.",
        "Review the PARTITION BY and ORDER BY columns and use a reliable ordering column such as a timestamp plus a tie-breaker, then validate that only one row per entity is returned.",
        ["Latest record logic is wrong", "Incorrect window function partition"],
    ),
    (
        "single",
        "The distinct customer count changes depending on which chart I look at.",
        "Different visuals may apply different filter context or dimensions that change the distinct-count population.",
        ["Distinct count changes across visuals"],
    ),
    (
        "single",
        "Two dashboards report different active users even though they use the same formula.",
        "Confirm both derive the metric from the same source tables, or from sources with equivalent business logic, and that the KPI definition matches.",
        ["Metric uses different source tables", "KPI definition mismatch"],
    ),
    (
        "single",
        "Lots of sales rows in the warehouse are assigned to an 'Unknown' customer.",
        "Check whether fact records match the expected dimension keys and how missing dimension members are handled; look for fact rows whose foreign keys have no matching dimension record.",
        ["Dimension key mismatch", "Missing foreign key relationships"],
    ),
    (
        "single",
        "Our sales fact table has the same order_id and product_id combination several times.",
        "Identify the intended grain of the fact table and verify that each key combination appears only once at that grain.",
        ["Fact table grain duplication", "Duplicate records"],
    ),
    # --- ambiguous wording (known retrieval trap) -----------------------------------
    (
        "ambiguous",
        "My conversion rate is above 100%.",
        "Check for duplicate numerator records, overlapping categories, or a denominator filtered more narrowly than the numerator. Validate the denominator separately and confirm both use the same population, filters and period.",
        ["Percentage above 100 percent", "Wrong KPI denominator"],
    ),
    # --- multi-issue ------------------------------------------------------------------
    (
        "multi",
        "Revenue on my Power BI report is higher than finance's and some orders look duplicated.",
        "Confirm both use the same revenue definition (refunds, cancellations, discounts, pending payments). Check for duplicate rows or duplicate source transactions and one-to-many joins, and review Power BI filters, relationships and DAX measures.",
        [
            "Revenue does not match finance report",
            "Duplicate records",
            "Duplicate source transactions",
            "Power BI dashboard mismatch",
            "Dashboard total is higher than the source system",
        ],
    ),
    # --- out of scope: the assistant should not invent an answer ---------------------
    (
        "out_of_scope",
        "How do I bake sourdough bread?",
        "This question is outside the troubleshooting knowledge base; the assistant should say it needs more information or that the topic is not covered.",
        [],
    ),
    (
        "out_of_scope",
        "What is the best laptop for gaming?",
        "This question is outside the troubleshooting knowledge base; the assistant should say it needs more information or that the topic is not covered.",
        [],
    ),
    (
        "out_of_scope",
        "Write me a Python script that sends emails.",
        "This question is outside the troubleshooting knowledge base; the assistant should say it needs more information or that the topic is not covered.",
        [],
    ),
]


def main():
    kb_issues = {
        item["issue"]
        for item in json.loads(Path("data/troubleshooting_guide.json").read_text(encoding="utf-8"))
    }
    out = Path("evals/golden_dataset.jsonl")
    with out.open("w", encoding="utf-8") as f:
        for n, (category, question, reference, relevant) in enumerate(ITEMS, start=1):
            unknown = set(relevant) - kb_issues
            assert not unknown, f"q{n:02d}: not in knowledge base: {unknown}"
            row = {
                "id": f"q{n:02d}",
                "category": category,
                "question": question,
                "reference": reference,
                "relevant_issues": relevant,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(ITEMS)} items to {out}")


if __name__ == "__main__":
    main()
