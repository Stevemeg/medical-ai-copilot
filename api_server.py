"""
FastAPI server exposing the existing RAG pipeline (backend/rag_pipeline.py)
over HTTP, so a real, custom-built frontend (frontend/index.html) can call
it directly via fetch(), instead of being constrained to Streamlit's
component model.

Run with:
    uvicorn api_server:app --reload --port 8000

The existing Streamlit app (app.py) is untouched and still works
independently -- this is an additional way to run the same backend logic,
not a replacement for it.
"""

import re
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backend.rag_pipeline import answer_question

app = FastAPI(title="Medical AI Copilot API")

# CORS: wide open deliberately. This is a local-dev / portfolio deployment
# with no patient data and no auth -- not a multi-tenant production
# service. If this is ever deployed somewhere with real users or data,
# restrict allow_origins to the actual frontend's domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------
# Source name mapping (same mapping used previously in the Streamlit app,
# now living here since the frontend is plain JS and shouldn't duplicate
# this Python-side parsing logic)
# -----------------------------------
SOURCE_DISPLAY_NAMES = {
    "nice_diabetic_foot_guideline.pdf.pdf.txt": "NICE NG19 — Diabetic Foot Problems",
    "nice_hypertension_guideline.pdf.pdf.txt": "NICE NG136 — Hypertension",
    "nice_cvd_lipid_guideline.pdf.pdf.txt": "NICE NG238 — Cardiovascular Risk & Lipids",
    "nice_type2_diabetes_guideline.pdf.txt": "NICE NG28 — Type 2 Diabetes",
    "moh_diabetes_mellitus_guideline.pdf.txt": "MoH — Diabetes Mellitus Guideline",
    "who_doc_1.pdf.txt": "WHO — Tuberculosis Report",
    "who_doc_2.pdf.txt": "WHO — Malaria Report",
    "cdc_chronic_disease_overview.pdf.txt": "CDC — Chronic Disease Overview",
    "openstax_anatomy_physiology.pdf.txt": "OpenStax — Anatomy & Physiology",
}

EXAMPLE_QUESTIONS = [
    "How should a diabetic foot ulcer be managed?",
    "When should statins be offered for cardiovascular risk reduction?",
    "What is the SINBAD classification?",
    "Explain insulin resistance.",
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

    return [
        {"name": name, "pages": sorted(grouped[name], key=page_sort_key)}
        for name in order
    ]


class AskRequest(BaseModel):
    question: str


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/examples")
def examples():
    return {"questions": EXAMPLE_QUESTIONS}


@app.post("/api/ask")
def ask(req: AskRequest):
    result = answer_question(req.question)
    return {
        "answer": result["answer"],
        "sources": group_sources(result["sources"]),
    }