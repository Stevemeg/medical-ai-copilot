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
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backend.knowledge_models import RetrievalPolicy
from backend.source_registry import SourceRegistry
from backend.patient_api import router as patient_router

app = FastAPI(title="Medical AI Copilot API")
app.include_router(patient_router)

# CORS: wide open deliberately. This is a local-dev / portfolio deployment
# with synthetic patient data and no auth -- not a multi-tenant production
# service. If this is ever deployed somewhere with real users or data,
# restrict allow_origins to the actual frontend's domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    }
