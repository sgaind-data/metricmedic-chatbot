import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import errors


load_dotenv()


class MetricMedicLLM:
    """Gemini wrapper for grounded troubleshooting responses."""

    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY was not found. "
                "Add it to a .env file before running MetricMedic."
            )

        self.client = genai.Client(api_key=api_key)

        # Primary model used for generation.
        self.model_name = "gemini-2.5-flash"

    @staticmethod
    def _build_context(retrieved_results):
        """Convert retrieved troubleshooting records into grounded context."""
        context_blocks = []

        for i, result in enumerate(retrieved_results, start=1):
            context_blocks.append(
                (
                    f"Context {i}\n"
                    f"Issue: {result['issue']}\n"
                    f"Keywords: {result['keywords']}\n"
                    f"Guidance: {result['response']}\n"
                    f"Similarity score: {result['score']:.4f}"
                )
            )

        return "\n\n".join(context_blocks)

    @staticmethod
    def _fallback_answer(retrieved_results):
        """
        Return a deterministic grounded response if Gemini
        is temporarily unavailable.
        """
        if not retrieved_results:
            return (
                "I couldn't find enough relevant troubleshooting context. "
                "Please describe the issue in more detail."
            )

        top_result = retrieved_results[0]

        related_issues = [
            result["issue"]
            for result in retrieved_results[1:3]
        ]

        answer = (
            "### Likely issue\n"
            f"The most relevant troubleshooting match is "
            f"**{top_result['issue']}**.\n\n"
            "### Recommended checks\n"
            f"- {top_result['response']}\n"
        )

        for result in retrieved_results[1:3]:
            answer += f"- Also review: {result['response']}\n"

        answer += (
            "\n### Why this may be happening\n"
            "The retrieved troubleshooting knowledge suggests that this issue "
            "may be related to the conditions above. Review the recommended "
            "checks to isolate the root cause."
        )

        if related_issues:
            answer += (
                "\n\nRelated retrieved issues: "
                + ", ".join(related_issues)
                + "."
            )

        return answer

    def generate_answer(self, user_query, retrieved_results):
        """
        Generate a concise troubleshooting response grounded in
        retrieved MetricMedic knowledge-base entries.
        """

        if not retrieved_results:
            return (
                "I couldn't find enough relevant troubleshooting context "
                "for that issue. Try describing the SQL, dashboard, KPI, "
                "filter, join, refresh, or data-quality problem in more detail."
            )

        context = self._build_context(retrieved_results)

        prompt = f"""
You are MetricMedic, an AI troubleshooting assistant for students
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
say that more information is needed instead of guessing.

USER QUESTION:
{user_query}

RETRIEVED TROUBLESHOOTING CONTEXT:
{context}
"""

        max_attempts = 3

        for attempt in range(max_attempts):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                )

                if response.text:
                    return response.text.strip()

            except errors.ServerError as error:
                if attempt < max_attempts - 1:
                    wait_time = 2 ** attempt

                    print(
                        f"Gemini temporarily unavailable. "
                        f"Retrying in {wait_time} seconds..."
                    )

                    time.sleep(wait_time)

                else:
                    print(
                        "Gemini remained unavailable. "
                        "Using retrieved-context fallback."
                    )

            except Exception as error:
                print(f"Gemini generation error: {error}")
                break

        return self._fallback_answer(retrieved_results)