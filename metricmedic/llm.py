"""Gemini chat model via LangChain (replaces the hand-rolled google-genai client)."""

from langchain_google_genai import ChatGoogleGenerativeAI

from metricmedic.config import GEMINI_MODEL, GEMINI_TEMPERATURE, get_gemini_api_key


def get_chat_model(model: str = GEMINI_MODEL) -> ChatGoogleGenerativeAI:
    """Return a Gemini chat model.

    max_retries replaces the manual exponential-backoff loop in the old llm.py.
    Kept low on purpose so free-tier quota isn't burned on retries.
    """
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=get_gemini_api_key(),
        temperature=GEMINI_TEMPERATURE,
        max_retries=2,
    )
