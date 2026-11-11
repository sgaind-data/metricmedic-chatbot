"""Request/response models. These define the API contract shown at /docs."""

from pydantic import BaseModel, Field

from metricmedic.pipeline import MAX_QUERY_CHARS, MAX_TOP_K, MIN_TOP_K


class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=MAX_QUERY_CHARS,
        description="A SQL, dashboard, KPI, ETL or data-quality problem in plain English.",
        examples=["Why did records disappear after an INNER JOIN?"],
    )
    top_k: int = Field(
        3,
        ge=MIN_TOP_K,
        le=MAX_TOP_K,
        description="How many knowledge-base scenarios to retrieve.",
    )
    session_id: str | None = Field(
        None,
        max_length=128,
        description="Optional conversation ID; groups related questions in Langfuse.",
    )


class Source(BaseModel):
    issue: str
    keywords: list[str] | str
    response: str
    score: float = Field(..., description="Cosine similarity, higher is more relevant.")
    kb_index: int


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    used_fallback: bool = Field(
        ..., description="True when Gemini failed and the answer was built from sources only."
    )
    latency_ms: float
    request_id: str


class HealthResponse(BaseModel):
    status: str
    version: str
    kb_entries: int


class ErrorResponse(BaseModel):
    detail: str
    request_id: str | None = None
