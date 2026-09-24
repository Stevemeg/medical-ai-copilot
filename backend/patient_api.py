"""Versioned API for synthetic patient import and reproducible reviews."""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException
from pydantic import BaseModel, RootModel

from backend.fhir_adapter import FHIRInputError, parse_bundle
from backend.patient_context import data_availability, timeline
from backend.patient_models import ClinicalReview, DataAvailability, PatientContext, TimelineEvent
from backend.patient_store import PatientNotFound, ReviewNotFound, SQLitePatientRepository
from backend.patient_store import FindingNotFound, ReviewCompleted
from backend.clinical_models import ActionRequest, ClinicalFinding, EvaluationRequest, FindingAction
from backend.clinical_review import ClinicalReviewEngine, EvaluationConflict

router = APIRouter(prefix="/v1")
log = logging.getLogger(__name__)
FIXTURES = Path(__file__).resolve().parents[1] / "data" / "synthetic_fhir"
DEMO_LABELS = {
    "syn_pat_001": "Synthetic Patient A - Type 2 diabetes",
    "syn_pat_002": "Synthetic Patient B - Hypertension",
    "syn_pat_003": "Synthetic Patient C - Combined context",
    "syn_pat_004": "Synthetic Patient D - Incomplete record",
    "syn_pat_005": "Synthetic Patient E - Medication and allergy",
}


def get_repository() -> SQLitePatientRepository:
    default_path = Path(__file__).resolve().parents[1] / "data" / "patient_context.db"
    return SQLitePatientRepository(os.environ.get("PATIENT_DB_PATH", str(default_path)))


class ValidationResponse(BaseModel):
    valid: bool
    resource_counts: dict[str, int]
    warnings: list[str]
    errors: list[dict[str, str]]


class FHIRBundleInput(RootModel[dict[str, Any]]):
    """Raw JSON object at the interoperability boundary; adapter validates its fields."""


class ImportResponse(BaseModel):
    patient_id: str
    source_patient_id: str
    context_hash: str
    resource_counts: dict[str, int]
    warnings: list[str]
    changed: bool


class PatientListItem(BaseModel):
    patient_id: str
    source_patient_id: str
    synthetic_label: str
    context_hash: str


class PatientResponse(BaseModel):
    patient_id: str
    context_hash: str
    context: PatientContext
    timeline: tuple[TimelineEvent, ...]
    data_availability: DataAvailability


def _api_error(status: int, code: str, message: str, path: str | None = None) -> HTTPException:
    detail: dict[str, str] = {"code": code, "message": message}
    if path:
        detail["path"] = path
    return HTTPException(status_code=status, detail=detail)


def _storage_error(exc: Exception) -> HTTPException:
    log.error("patient_storage_failure type=%s", type(exc).__name__)
    return _api_error(503, "storage_error", "Patient storage is unavailable")


def _import(raw: Any, repository: SQLitePatientRepository) -> ImportResponse:
    try:
        context, counts = parse_bundle(raw)
    except FHIRInputError as exc:
        raise _api_error(422, exc.code, exc.message, exc.path) from exc
    try:
        patient_id, digest, changed = repository.import_context(context)
    except (sqlite3.Error, ValueError) as exc:
        raise _storage_error(exc) from exc
    log.info(
        "patient_import patient_id=%s resource_count=%d context_hash=%s changed=%s",
        patient_id,
        sum(counts.values()),
        digest,
        changed,
    )
    return ImportResponse(
        patient_id=patient_id,
        source_patient_id=context.patient.source_patient_id,
        context_hash=digest,
        resource_counts=counts,
        warnings=[],
        changed=changed,
    )


@router.post("/fhir/validate", response_model=ValidationResponse)
def validate_fhir(raw: FHIRBundleInput = Body(...)) -> ValidationResponse:
    try:
        _, counts = parse_bundle(raw.root)
    except FHIRInputError as exc:
        return ValidationResponse(valid=False, resource_counts={}, warnings=[], errors=[exc.as_dict()])
    return ValidationResponse(valid=True, resource_counts=counts, warnings=[], errors=[])


@router.post("/patients/import", response_model=ImportResponse)
def import_patient(
    raw: FHIRBundleInput = Body(...), repository: SQLitePatientRepository = Depends(get_repository)
) -> ImportResponse:
    return _import(raw.root, repository)


@router.get("/demo-patients")
def demo_patients() -> list[dict[str, str]]:
    return [{"key": key, "label": label} for key, label in DEMO_LABELS.items()]


@router.post("/demo-patients/{key}/load", response_model=ImportResponse)
def load_demo(key: str, repository: SQLitePatientRepository = Depends(get_repository)) -> ImportResponse:
    if key not in DEMO_LABELS:
        raise _api_error(404, "demo_not_found", "Demo patient not found")
    raw = json.loads((FIXTURES / f"{key}.json").read_text(encoding="utf-8"))
    return _import(raw, repository)


@router.get("/patients", response_model=list[PatientListItem])
def list_patients(repository: SQLitePatientRepository = Depends(get_repository)) -> list[PatientListItem]:
    try:
        return [
            PatientListItem(
                patient_id=pid,
                source_patient_id=context.patient.source_patient_id,
                synthetic_label=context.patient.synthetic_label,
                context_hash=digest,
            )
            for pid, context, digest in repository.list_patients()
        ]
    except (sqlite3.Error, ValueError) as exc:
        raise _storage_error(exc) from exc


@router.get("/patients/{patient_id}", response_model=PatientResponse)
def get_patient(patient_id: str, repository: SQLitePatientRepository = Depends(get_repository)) -> PatientResponse:
    try:
        context, digest = repository.get_patient(patient_id)
    except PatientNotFound as exc:
        raise _api_error(404, "patient_not_found", "Patient not found") from exc
    except (sqlite3.Error, ValueError) as exc:
        raise _storage_error(exc) from exc
    return PatientResponse(
        patient_id=patient_id,
        context_hash=digest,
        context=context,
        timeline=timeline(context),
        data_availability=data_availability(context),
    )


@router.post("/patients/{patient_id}/reviews", response_model=ClinicalReview, status_code=201)
def create_review(patient_id: str, repository: SQLitePatientRepository = Depends(get_repository)) -> ClinicalReview:
    try:
        review = repository.create_review(patient_id)
    except PatientNotFound as exc:
        raise _api_error(404, "patient_not_found", "Patient not found") from exc
    except (sqlite3.Error, ValueError) as exc:
        raise _storage_error(exc) from exc
    log.info(
        "review_created review_id=%s patient_id=%s context_hash=%s",
        review.review_id,
        patient_id,
        review.patient_context_hash,
    )
    return review


@router.get("/patients/{patient_id}/reviews", response_model=list[ClinicalReview])
def list_reviews(
    patient_id: str, repository: SQLitePatientRepository = Depends(get_repository)
) -> list[ClinicalReview]:
    try:
        return repository.list_reviews(patient_id)
    except PatientNotFound as exc:
        raise _api_error(404, "patient_not_found", "Patient not found") from exc
    except (sqlite3.Error, ValueError) as exc:
        raise _storage_error(exc) from exc


@router.get("/reviews/{review_id}", response_model=ClinicalReview)
def get_review(review_id: str, repository: SQLitePatientRepository = Depends(get_repository)) -> ClinicalReview:
    try:
        return repository.get_review(review_id)
    except ReviewNotFound as exc:
        raise _api_error(404, "review_not_found", "Review not found") from exc
    except (sqlite3.Error, ValueError) as exc:
        raise _storage_error(exc) from exc


@router.post("/reviews/{review_id}/complete", response_model=ClinicalReview)
def complete_review(review_id: str, repository: SQLitePatientRepository = Depends(get_repository)) -> ClinicalReview:
    try:
        return repository.complete_review(review_id)
    except ReviewNotFound as exc:
        raise _api_error(404, "review_not_found", "Review not found") from exc
    except (sqlite3.Error, ValueError) as exc:
        raise _storage_error(exc) from exc


@router.post("/reviews/{review_id}/evaluate", response_model=tuple[ClinicalFinding, ...])
def evaluate_review(
    review_id: str, request: EvaluationRequest, repository: SQLitePatientRepository = Depends(get_repository)
) -> tuple[ClinicalFinding, ...]:
    try:
        return ClinicalReviewEngine(repository).evaluate(review_id, request)
    except ReviewNotFound as exc:
        raise _api_error(404, "review_not_found", "Review not found") from exc
    except ReviewCompleted as exc:
        raise _api_error(409, "review_completed", "Completed review cannot be evaluated") from exc
    except EvaluationConflict as exc:
        raise _api_error(409, "evaluation_conflict", str(exc)) from exc
    except (sqlite3.Error, ValueError) as exc:
        raise _storage_error(exc) from exc


@router.get("/reviews/{review_id}/findings", response_model=tuple[ClinicalFinding, ...])
def list_findings(
    review_id: str, repository: SQLitePatientRepository = Depends(get_repository)
) -> tuple[ClinicalFinding, ...]:
    try:
        return repository.list_findings(review_id)
    except ReviewNotFound as exc:
        raise _api_error(404, "review_not_found", "Review not found") from exc
    except (sqlite3.Error, ValueError) as exc:
        raise _storage_error(exc) from exc


@router.get("/findings/{finding_id}", response_model=ClinicalFinding)
def get_finding(finding_id: str, repository: SQLitePatientRepository = Depends(get_repository)) -> ClinicalFinding:
    try:
        return repository.get_finding(finding_id)
    except FindingNotFound as exc:
        raise _api_error(404, "finding_not_found", "Finding not found") from exc
    except (sqlite3.Error, ValueError) as exc:
        raise _storage_error(exc) from exc


@router.post("/findings/{finding_id}/actions", response_model=FindingAction, status_code=201)
def add_finding_action(
    finding_id: str, request: ActionRequest, repository: SQLitePatientRepository = Depends(get_repository)
) -> FindingAction:
    try:
        return repository.add_action(finding_id, request.action_type, request.note)
    except FindingNotFound as exc:
        raise _api_error(404, "finding_not_found", "Finding not found") from exc
    except (sqlite3.Error, ValueError) as exc:
        raise _storage_error(exc) from exc


@router.get("/findings/{finding_id}/actions", response_model=tuple[FindingAction, ...])
def list_finding_actions(
    finding_id: str, repository: SQLitePatientRepository = Depends(get_repository)
) -> tuple[FindingAction, ...]:
    try:
        return repository.list_actions(finding_id)
    except FindingNotFound as exc:
        raise _api_error(404, "finding_not_found", "Finding not found") from exc
    except (sqlite3.Error, ValueError) as exc:
        raise _storage_error(exc) from exc
