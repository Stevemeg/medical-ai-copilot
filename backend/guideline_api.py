"""Authenticated governance API. State and audit commit together."""

from dataclasses import asdict
from uuid import UUID

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy import select

from backend.db import GuidelineActivation, GuidelineCandidate, session_factory
from backend.guideline_updates import CandidateInput, decide, prepare, register_candidate
from backend.source_registry import SourceRegistry
from backend.settings import get_settings

router = APIRouter(prefix="/v1")


def enabled():
    if not get_settings().guideline_updates_enabled:
        raise HTTPException(503, "Guideline update storage is not enabled")


@router.get("/guidelines")
def guidelines():
    registry = SourceRegistry()
    return [{**asdict(doc), "versions": versions(doc.document_id)} for doc in registry.documents.values()]


@router.get("/guidelines/{document_id}/versions")
def versions(document_id: str):
    registry = SourceRegistry()
    if document_id not in registry.documents:
        raise HTTPException(404, "Unknown document")
    items = [asdict(v) for v in registry.versions.values() if v.document_id == document_id]
    if get_settings().guideline_updates_enabled:
        with session_factory()() as session:
            active = session.get(GuidelineActivation, document_id)
            if active:
                for item in items:
                    if item["status"] == "current":
                        item["status"] = "superseded"
            for candidate in session.scalars(
                select(GuidelineCandidate).where(
                    GuidelineCandidate.document_id == document_id, GuidelineCandidate.status == "activated"
                )
            ):
                items.append(
                    {
                        "version_id": candidate.version_id,
                        "sha256": candidate.checksum,
                        "status": "current"
                        if active and active.candidate_id == candidate.candidate_id
                        else "superseded",
                    }
                )
    return items


@router.get("/guideline-updates")
def updates():
    enabled()
    with session_factory()() as session:
        return [
            {
                "candidate_id": str(c.candidate_id),
                "document_id": c.document_id,
                "version_id": c.version_id,
                "base_version_id": c.base_version_id,
                "status": c.status,
                "checksum": c.checksum,
                "detected_at": c.created_at.isoformat(),
                "diff": c.diff,
            }
            for c in session.scalars(
                select(GuidelineCandidate).order_by(GuidelineCandidate.created_at.desc()).limit(100)
            )
        ]


@router.post("/guideline-updates", status_code=201)
def register(body: CandidateInput, request: Request):
    enabled()
    try:
        return {
            "candidate_id": register_candidate(body, request.state.actor, request.state.request_id),
            "status": "quarantined",
        }
    except PermissionError:
        raise HTTPException(403, "Authenticated editor required") from None
    except ValueError:
        raise HTTPException(422, "Candidate identity, source or content rejected") from None


class Decision(BaseModel):
    note: str = Field(min_length=5, max_length=1000)


@router.post("/guideline-updates/{candidate_id}/prepare")
def prepare_candidate(candidate_id: UUID, request: Request):
    enabled()
    try:
        prepare(str(candidate_id), request.state.actor, request.state.request_id)
    except PermissionError:
        raise HTTPException(403, "Authenticated editor required") from None
    except ValueError:
        raise HTTPException(409, "Candidate preparation rejected") from None
    return {"status": "review_required"}


@router.post("/guideline-updates/{candidate_id}/{action}")
def decision(candidate_id: UUID, action: str, body: Decision, request: Request):
    enabled()
    try:
        return decide(str(candidate_id), action, body.note, request.state.actor, request.state.request_id)
    except PermissionError:
        raise HTTPException(403, "Authenticated editor required") from None
    except ValueError:
        raise HTTPException(409, "Governance transition rejected; check state, diff and base version") from None
