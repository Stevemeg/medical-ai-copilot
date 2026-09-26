"""
FastAPI server exposing the existing RAG pipeline (backend/rag_pipeline.py)
over HTTP, so a real, custom-built frontend (frontend/index.html) can call
it directly via fetch(), instead of being constrained to Streamlit's
component model.

Run with:
    uvicorn api_server:app --reload --port 8000

The Streamlit app uses the same governed retrieval path.
"""

import re
import sys
import hashlib
import logging
import time
from uuid import UUID, uuid4
from pathlib import Path

from fastapi import FastAPI, Request, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import text as sql_text
from sqlalchemy import select
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backend.knowledge_models import RetrievalPolicy
from backend.evidence_models import EvidenceQueryRequest, EvidenceQueryResponse
from backend.evidence_pipeline import public_answer
from backend.source_registry import SourceRegistry
from backend.patient_api import router as patient_router
from backend.db import engine, session_factory, AuditEvent
from backend.settings import get_settings
from backend.security import actor_from_request, required_roles
from backend.operations import audit_operation, rate_limit
from backend.index_provenance import validate_manifest, validate_chunks_manifest
from backend.logging_config import configure_logging
from backend.request_context import set_request_id, reset_request_id
from opentelemetry import trace

settings = get_settings()
configure_logging(settings.log_level, settings.app_env == "production")
logger = logging.getLogger("api")
HTTP_COUNT = Counter("medical_http_requests_total", "Requests", ["method", "route", "status"])
HTTP_LATENCY = Histogram("medical_http_latency_seconds", "Request latency", ["method", "route"])
EVIDENCE_COUNT = Counter("medical_evidence_queries_total", "Evidence queries", ["status"])
DB_POOL = Gauge("medical_db_pool_connections", "Database pool connections", ["state"])

app = FastAPI(title="Medical AI Copilot API")
app.include_router(patient_router)


@app.exception_handler(HTTPException)
async def structured_http_error(request: Request, exc: HTTPException):
    detail = exc.detail if isinstance(exc.detail, dict) else {"code": "request_error", "message": str(exc.detail)}
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": detail, "error": {**detail, "request_id": getattr(request.state, "request_id", "")}},
    )


@app.middleware("http")
async def operational_boundary(request: Request, call_next):
    started = time.perf_counter()
    path = request.url.path
    supplied = request.headers.get("x-request-id", "")
    try:
        request_id = str(UUID(supplied))
    except ValueError:
        request_id = str(uuid4())
    request.state.request_id = request_id
    request_token = set_request_id(request_id)
    route = "/v1/protected" if path.startswith("/v1/") else "/unmatched"
    actor = None
    try:
        if path.startswith("/v1/patients/import") or path.startswith("/v1/fhir/validate"):
            body = await request.body()
            if len(body) > settings.max_fhir_body_bytes:
                raise HTTPException(413, detail={"code": "body_too_large", "message": "FHIR body exceeds size limit"})
        roles = required_roles(path, request.method)
        if roles is not None:
            actor = actor_from_request(request)
            request.state.actor = actor
            if not actor.roles.intersection(roles):
                if settings.app_env != "test":
                    audit_operation("authorization_denied", actor, request_id, payload={"route": route})
                raise HTTPException(403, detail={"code": "forbidden", "message": "Permission denied"})
            operation = None
            if path in {"/v1/evidence/query", "/api/ask"}:
                operation = "evidence_query"
            elif path == "/v1/patients/import" or (path.startswith("/v1/demo-patients/") and path.endswith("/load")):
                operation = "patient_import"
            elif path.endswith("/evaluate"):
                operation = "review_evaluation"
            if operation and settings.app_env != "test" and not rate_limit(actor, operation):
                raise HTTPException(429, detail={"code": "rate_limited", "message": "Rate limit exceeded"})
        with trace.get_tracer(__name__).start_as_current_span("http.request") as span:
            span.set_attribute("http.method", request.method)
            span.set_attribute("request.id", request_id)
            response = await call_next(request)
        if path in {"/v1/evidence/query", "/api/ask"} and response.status_code < 500:
            EVIDENCE_COUNT.labels("executed" if response.status_code < 400 else "rejected").inc()
            if settings.app_env != "test" and actor is not None:
                # The query is deliberately omitted. Only request metadata is persisted.
                audit_operation(
                    "evidence_query_executed",
                    actor,
                    request_id,
                    "evidence_query",
                    request_id,
                    {"status_code": response.status_code},
                )
    except HTTPException as exc:
        if exc.status_code == 401 and settings.app_env != "test":
            from backend.security import ActorContext

            audit_operation(
                "authentication_failed",
                ActorContext("unknown", frozenset(), "unknown", False),
                request_id,
                payload={"route": route},
            )
        detail = exc.detail if isinstance(exc.detail, dict) else {"code": "request_error", "message": str(exc.detail)}
        response = JSONResponse(
            status_code=exc.status_code, content={"detail": detail, "error": {**detail, "request_id": request_id}}
        )
    except Exception as exc:
        logger.error(
            "request_failed", extra={"request_id": request_id, "route": route, "exception_type": type(exc).__name__}
        )
        response = JSONResponse(
            status_code=503,
            content={
                "error": {"code": "service_unavailable", "message": "Service unavailable", "request_id": request_id}
            },
        )
    route = getattr(request.scope.get("route"), "path", route)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'"
    )
    latency = time.perf_counter() - started
    HTTP_COUNT.labels(request.method, route, str(response.status_code)).inc()
    HTTP_LATENCY.labels(request.method, route).observe(latency)
    logger.info(
        "http_request",
        extra={
            "request_id": request_id,
            "route": route,
            "method": request.method,
            "status_code": response.status_code,
            "latency_ms": round(latency * 1000, 2),
            "actor_subject": hashlib.sha256(actor.subject.encode()).hexdigest()[:16] if actor else None,
        },
    )
    reset_request_id(request_token)
    return response


@app.exception_handler(RequestValidationError)
async def phase3_validation_error(request: Request, exc: RequestValidationError):
    path = request.url.path
    if path.startswith("/v1/reviews/") and path.endswith("/evaluate"):
        detail = {"code": "invalid_evaluation_context", "message": "Evaluation context is invalid"}
        return JSONResponse(
            status_code=422,
            content={"detail": detail, "error": {**detail, "request_id": getattr(request.state, "request_id", "")}},
        )
    if path.startswith("/v1/findings/") and path.endswith("/actions"):
        detail = {"code": "invalid_action", "message": "Finding action is invalid"}
        return JSONResponse(
            status_code=422,
            content={"detail": detail, "error": {**detail, "request_id": getattr(request.state, "request_id", "")}},
        )
    return JSONResponse(
        status_code=422,
        content={
            "detail": exc.errors(),
            "error": {
                "code": "validation_error",
                "message": "Invalid request",
                "request_id": getattr(request.state, "request_id", ""),
            },
        },
    )


app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["Authorization", "Content-Type", "Idempotency-Key", "X-Request-ID"],
)


@app.get("/health/live")
def live():
    return {"status": "ok"}


@app.get("/health/ready")
def ready():
    try:
        with engine().connect() as connection:
            connection.execute(sql_text("SELECT 1"))
        validate_chunks_manifest(Path("data/chunks.json"), _registry)
        for label in ("clinical", "anatomy"):
            validate_manifest(
                Path(f"data/vector_store/{label}_faiss.index"),
                Path(f"data/vector_store/{label}_metadata.json"),
                _registry,
            )
    except Exception:
        raise HTTPException(503, detail={"code": "not_ready", "message": "Critical dependency unavailable"}) from None
    return {"status": "ready"}


@app.get("/metrics")
def metrics():
    pool = engine().pool
    DB_POOL.labels("checked_out").set(pool.checkedout())
    DB_POOL.labels("checked_in").set(pool.checkedin())
    DB_POOL.labels("overflow").set(pool.overflow())
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/v1/audit/events")
def audit_events(limit: int = 50, after: int = 0):
    if not 1 <= limit <= 100 or after < 0:
        raise HTTPException(422, detail={"code": "invalid_pagination", "message": "Invalid audit pagination"})
    with session_factory()() as session:
        events = session.scalars(
            select(AuditEvent)
            .where(AuditEvent.sequence_number > after)
            .order_by(AuditEvent.sequence_number)
            .limit(limit)
        ).all()
        return [
            {
                "sequence_number": e.sequence_number,
                "occurred_at": e.occurred_at.isoformat(),
                "actor_subject": e.actor_subject,
                "event_type": e.event_type,
                "resource_type": e.resource_type,
                "resource_id": e.resource_id,
                "request_id": e.request_id,
                "event_payload": e.event_payload,
            }
            for e in events
        ]


# -----------------------------------
# Source names come from the registry; legacy filename labels remain accepted.
# -----------------------------------
_registry = SourceRegistry()
SOURCE_DISPLAY_NAMES = {v.legacy_source: _registry.display_name(v) for v in _registry.versions.values()}


EXAMPLE_QUESTIONS = [
    "How should a diabetic foot ulcer be managed?",
    "When should statins be offered for cardiovascular risk reduction?",
    "What is the SINBAD classification?",
    "How should hypertension be diagnosed?",
]


def clean_fallback_name(raw_source: str) -> str:
    name = raw_source.split(",")[0]
    name = re.sub(r"\.pdf(\.pdf)?\.txt$", "", name)
    name = name.replace("_", " ").strip()
    return name.title()


def parse_source_entry(raw_entry: str):
    if "," in raw_entry:
        filename, page_part = raw_entry.split(",", 1)
        page_part = page_part.strip()
    else:
        filename, page_part = raw_entry, ""
    display_name = SOURCE_DISPLAY_NAMES.get(filename, clean_fallback_name(filename))
    page_number = re.sub(r"^pages?\s*", "", page_part).strip()
    return display_name, page_number


def group_sources(raw_sources: list[str]) -> list[dict]:
    """
    Groups raw source strings by document, combining page numbers into one
    entry per document instead of one entry per cited chunk. Mirrors the
    grouping logic verified in app.py's render_source_chips, now producing
    structured data for the frontend to render however it wants.
    """
    grouped: dict[str, list[str]] = {}
    order: list[str] = []
    for raw in raw_sources:
        name, page = parse_source_entry(raw)
        if name not in grouped:
            grouped[name] = []
            order.append(name)
        if page and page not in grouped[name]:
            grouped[name].append(page)

    def page_sort_key(page_str: str) -> int:
        match = re.search(r"\d+", page_str)
        return int(match.group()) if match else 0

    return [{"name": name, "pages": sorted(grouped[name], key=page_sort_key)} for name in order]


class AskRequest(BaseModel):
    question: str
    policy: RetrievalPolicy = RetrievalPolicy.CURRENT_CLINICAL


@app.post("/v1/evidence/query", response_model=EvidenceQueryResponse)
def evidence_query(req: EvidenceQueryRequest) -> dict:
    from backend.rag_pipeline import get_service

    return public_answer(get_service().answer(req.retrieval_request()))


def answer_question(question: str, policy: RetrievalPolicy):
    # Load the retrieval model only for evidence requests, not patient API imports.
    from backend.rag_pipeline import answer_question as run

    return run(question, policy=policy)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/examples")
def examples():
    return {"questions": EXAMPLE_QUESTIONS}


@app.get("/api/sources")
def sources():
    names = [
        SOURCE_DISPLAY_NAMES[v.legacy_source]
        for v in _registry.versions.values()
        if _registry.eligible({"version_id": v.version_id}, RetrievalPolicy.CURRENT_CLINICAL)
    ]
    return {"sources": sorted(names)}


@app.get("/v1/knowledge-sources")
def knowledge_sources():
    return [
        {
            "title": _registry.documents[v.document_id].canonical_title,
            "publisher": _registry.documents[v.document_id].publisher,
            "source_type": _registry.documents[v.document_id].source_type,
            "lifecycle": v.status,
            "version": v.version_label,
        }
        for v in _registry.versions.values()
    ]


@app.post("/api/ask")
def ask(req: AskRequest):
    result = answer_question(req.question, policy=req.policy)
    return {
        "answer": result["answer"],
        "sources": group_sources(result["sources"]),
        "citations": result["citations"],
        "status": result.get("status", "abstained"),
        "claims": result.get("claims", []),
        "conflicts": result.get("conflicts", []),
        "evidence": result.get("evidence", []),
        "retrieval": result.get("retrieval", {}),
    }
