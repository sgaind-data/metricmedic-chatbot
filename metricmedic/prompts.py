"""Prompt template and context formatting (moved from llm.py)."""

from langchain_core.prompts import ChatPromptTemplate

SYSTEM_PROMPT = """You are MetricMedic, an AI troubleshooting assistant for students
and junior data analysts.

Your role is to help diagnose SQL, dashboard, KPI, reporting,
ETL, and data-quality problems.

IMPORTANT GROUNDING RULES:
1. Use the retrieved troubleshooting context below as the primary
source for your answer.
2. Do not invent database schemas, table names, business rules,
company policies, or technical details that were not provided.
3. If multiple retrieved issues are relevant, combine them into
one coherent explanation.
4. Do not claim certainty when several causes are possible.
5. Phrase conclusions as "likely", "possible", or "may" when appropriate.
6. Keep the response practical and concise.
7. Recommend diagnostic checks before suggesting major changes.
8. Do not mention internal prompts or retrieved context in the answer.

Structure your response exactly like this:

### Likely issue
Provide 1-2 concise sentences explaining the most likely cause or causes.

### Recommended checks
Provide 3-5 practical troubleshooting steps as bullet points.

### Why this may be happening
Provide one short paragraph explaining the technical reason.

If the retrieved context does not sufficiently support an answer,
say that more information is needed instead of guessing."""

HUMAN_PROMPT = """USER QUESTION:
{question}

RETRIEVED TROUBLESHOOTING CONTEXT:
{context}"""

# Sent as ONE human message on purpose: Gemini's v1 API rejects system
# messages ("Developer instruction is not enabled for api version v1").
# A single message works on every API version, matching the original app.
RAG_PROMPT = ChatPromptTemplate.from_messages([("human", SYSTEM_PROMPT + "\n\n" + HUMAN_PROMPT)])

NO_CONTEXT_ANSWER = (
    "I couldn't find enough relevant troubleshooting context "
    "for that issue. Try describing the SQL, dashboard, KPI, "
    "filter, join, refresh, or data-quality problem in more detail."
)


def format_context(sources: list[dict]) -> str:
    """Convert retrieved records into grounded context blocks."""
    blocks = []
    for i, s in enumerate(sources, start=1):
        blocks.append(
            f"Context {i}\n"
            f"Issue: {s['issue']}\n"
            f"Keywords: {s['keywords']}\n"
            f"Guidance: {s['response']}\n"
            f"Similarity score: {s['score']:.4f}"
        )
    return "\n\n".join(blocks)


def fallback_answer(sources: list[dict]) -> str:
    """Deterministic grounded answer used if Gemini is unavailable."""
    if not sources:
        return NO_CONTEXT_ANSWER

    top = sources[0]
    related = [s["issue"] for s in sources[1:3]]

    answer = (
        "### Likely issue\n"
        f"The most relevant troubleshooting match is **{top['issue']}**.\n\n"
        "### Recommended checks\n"
        f"- {top['response']}\n"
    )
    for s in sources[1:3]:
        answer += f"- Also review: {s['response']}\n"

    answer += (
        "\n### Why this may be happening\n"
        "The retrieved troubleshooting knowledge suggests that this issue "
        "may be related to the conditions above. Review the recommended "
        "checks to isolate the root cause."
    )
    if related:
        answer += "\n\nRelated retrieved issues: " + ", ".join(related) + "."
    return answer
