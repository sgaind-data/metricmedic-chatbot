"""MetricMedic REST API.

Run locally:   uvicorn metricmedic.api.main:app --reload
Docs:          http://localhost:8000/docs
"""

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse

from metricmedic import __version__
from metricmedic.api.schemas import AskRequest, AskResponse, ErrorResponse, HealthResponse
from metricmedic.logging_config import setup_logging
from metricmedic.pipeline import InvalidQueryError, MetricMedicRAG

logger = logging.getLogger("metricmedic.api")


def create_app(rag: MetricMedicRAG | None = None) -> FastAPI:
    """App factory. Tests pass a RAG built with a fake LLM; production builds the real one."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        setup_logging()
        # Load the embedding model + FAISS index ONCE at startup, not per request.
        app.state.rag = rag or MetricMedicRAG()
        logger.info("API started (version %s)", __version__, extra={"event": "api_startup"})
        yield
        logger.info("API shutting down", extra={"event": "api_shutdown"})

    app = FastAPI(
        title="MetricMedic RAG API",
        version=__version__,
        description="Retrieval-augmented troubleshooting for SQL, dashboards, KPIs and ETL.",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        # Reuse the caller's X-Request-ID if given, so a request can be traced across services.
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    @app.exception_handler(InvalidQueryError)
    async def invalid_query_handler(request: Request, exc: InvalidQueryError):
        return JSONResponse(
            status_code=422,
            content=ErrorResponse(
                detail=str(exc), request_id=getattr(request.state, "request_id", None)
            ).model_dump(),
        )

    @app.exception_handler(Exception)
    async def unhandled_handler(request: Request, exc: Exception):
        request_id = getattr(request.state, "request_id", None)
        logger.exception("Unhandled error", extra={"event": "api_error"})
        # Never leak internals (stack traces, keys) to the client.
        error = ErrorResponse(detail="Internal server error", request_id=request_id)
        return JSONResponse(status_code=500, content=error.model_dump())

    def get_rag(request: Request) -> MetricMedicRAG:
        return request.app.state.rag

    @app.get("/health", response_model=HealthResponse, tags=["ops"])
    def health(rag: MetricMedicRAG = Depends(get_rag)) -> HealthResponse:
        """Liveness/readiness: OK once the model and index are loaded. Makes no Gemini call."""
        return HealthResponse(
            status="ok", version=__version__, kb_entries=len(rag.retriever.knowledge_base)
        )

    @app.post(
        "/ask",
        response_model=AskResponse,
        responses={422: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
        tags=["rag"],
    )
    def ask(
        body: AskRequest, request: Request, rag: MetricMedicRAG = Depends(get_rag)
    ) -> AskResponse:
        """Retrieve relevant scenarios and generate a grounded answer (1 Gemini call)."""
        # Plain `def` (not async): the chain is blocking, so FastAPI runs it in a
        # worker thread and the server keeps accepting other requests.
        start = time.perf_counter()
        result = rag.answer(body.question, top_k=body.top_k)
        return AskResponse(
            answer=result["answer"],
            sources=result["sources"],
            used_fallback=result["used_fallback"],
            latency_ms=round((time.perf_counter() - start) * 1000, 1),
            request_id=request.state.request_id,
        )

    return app


app = create_app()
