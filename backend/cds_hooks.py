"""CDS Hooks 2.0.1 patient-view 1.0; deterministic informational cards."""

from datetime import datetime, timezone
from typing import Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, field_validator

from backend.clinical_models import EvaluationRequest
from backend.clinical_review import ClinicalReviewEngine
from backend.fhir_adapter import FHIR_ID, parse_bundle
from backend.fhir_client import FHIRClient, RESOURCE_TYPES
from backend.patient_api import get_repository
from backend.postgres_store import PostgresPatientRepository
from backend.safe_http import SafeHTTP, origin
from backend.settings import get_settings

router = APIRouter()
SERVICE_ID = "medical-patient-review"
PREFETCH = {
    "patient": "Patient/{{context.patientId}}",
    **{kind: f"{kind}?patient={{{{context.patientId}}}}&_count=100" for kind in RESOURCE_TYPES},
}


class HookContext(BaseModel):
    userId: str = Field(pattern=r"^(Practitioner|PractitionerRole|Patient|RelatedPerson)/[A-Za-z0-9.-]{1,64}$")
    patientId: str = Field(pattern=r"^[A-Za-z0-9.-]{1,64}$")
    encounterId: str | None = Field(default=None, pattern=r"^[A-Za-z0-9.-]{1,64}$")


class FHIRAuthorization(BaseModel):
    access_token: str = Field(min_length=1, max_length=8192, repr=False)
    token_type: Literal["Bearer"]
    expires_in: int = Field(gt=0)
    scope: str
    subject: str


class HookRequest(BaseModel):
    hook: Literal["patient-view"]
    hookInstance: UUID
    context: HookContext
    fhirServer: str | None = None
    fhirAuthorization: FHIRAuthorization | None = Field(default=None, repr=False)
    prefetch: dict | None = None

    @field_validator("fhirServer")
    @classmethod
    def registered_server(cls, value: str | None) -> str | None:
        if value is not None and value not in get_settings().smart_issuers:
            raise ValueError("FHIR server is not registered")
        return value


@router.get("/cds-services")
def discovery() -> dict:
    return {
        "services": [
            {
                "hook": "patient-view",
                "id": SERVICE_ID,
                "title": "Medical AI Copilot clinical review",
                "description": "Synthetic adult follow-up checks with governed evidence; informational only.",
                "prefetch": PREFETCH,
            }
        ]
    }


def hook_bundle(hook: HookRequest, http: SafeHTTP | None = None) -> dict:
    prefetch = hook.prefetch or {}
    resources = []
    complete = True
    for key in PREFETCH:
        item = prefetch.get(key)
        if item is None:
            complete = False
            break
        if key == "patient":
            if item.get("resourceType") != "Patient" or item.get("id") != hook.context.patientId:
                raise ValueError("Prefetch patient context mismatch")
            resources.append(item)
        else:
            if item.get("resourceType") != "Bundle" or item.get("type") != "searchset":
                raise ValueError("Prefetch must be a searchset")
            if any(link.get("relation") == "next" for link in item.get("link", [])):
                complete = False
                break
            entries = item.get("entry", [])
            if len(entries) > 100:
                raise ValueError("Prefetch limit exceeded")
            for entry in entries:
                resource = entry.get("resource", {})
                if resource.get("resourceType") != key:
                    raise ValueError("Unexpected prefetched resource")
                resources.append(resource)
    if complete:
        bundle = {"resourceType": "Bundle", "type": "collection", "entry": [{"resource": r} for r in resources]}
        context, _ = parse_bundle(bundle)
        if not FHIR_ID.fullmatch(context.patient.source_patient_id):
            raise ValueError("Invalid context")
        return bundle
    if hook.fhirServer and hook.fhirAuthorization:
        client = http or SafeHTTP({origin(hook.fhirServer)})
        return FHIRClient(hook.fhirServer, hook.fhirAuthorization.access_token, client).bundle(hook.context.patientId)
    raise ValueError("Complete supported prefetch or FHIR authorization required")


def cards(findings) -> list[dict]:
    result = []
    for finding in findings:
        if finding.status.value in {"satisfied", "not_applicable"}:
            continue
        detail = finding.rationale
        for ref in finding.evidence_refs:
            detail += (
                f"\n\n{ref.publisher}: {ref.guideline_title}; {ref.version_id or 'recommendation snapshot'}; "
                f"recommendation {ref.recommendation_id}; verified {ref.source_verified_on}."
            )
        card = {
            "uuid": str(UUID(finding.finding_id.removeprefix("finding-"))),
            "summary": f"{finding.title}: {finding.status.value.replace('_', ' ')}"[:140],
            "detail": detail,
            "indicator": "warning" if finding.status.value == "potential_care_gap" else "info",
            "source": {"label": "Medical AI Copilot deterministic rules"},
        }
        if finding.evidence_refs:
            card["source"]["url"] = finding.evidence_refs[0].canonical_source_url
        result.append(card)
    return result


def evaluation_context(context) -> EvaluationRequest:
    settings = get_settings()
    if settings.cds_synthetic_fixture_coverage and settings.app_env != "production":
        import json
        from backend.patient_api import FIXTURES
        from backend.patient_context import context_hash

        manifest = json.loads((FIXTURES / "manifest.json").read_text(encoding="utf-8"))["fixtures"]
        for key, row in manifest.items():
            fixture, _ = parse_bundle(json.loads((FIXTURES / f"{key}.json").read_text(encoding="utf-8")))
            if context_hash(fixture) == context_hash(context):
                return EvaluationRequest(as_of=row["evaluation_as_of"], record_coverage=row["record_coverage"])
    return EvaluationRequest(as_of=datetime.now(timezone.utc).date())


@router.post("/cds-services/{service_id}")
def patient_view(service_id: str, hook: HookRequest, repository: PostgresPatientRepository = Depends(get_repository)):
    if service_id != SERVICE_ID:
        raise HTTPException(404, "Unknown CDS service")
    try:
        bundle = hook_bundle(hook)
        context, _ = parse_bundle(bundle)
    except (ValueError, KeyError, TypeError, AttributeError):
        raise HTTPException(422, "Invalid or unavailable patient-view FHIR context") from None
    patient_id, _, _ = repository.import_context(context)
    review = (
        repository.create_review(patient_id, idempotency_key=f"cds-{hook.hookInstance}")
        if isinstance(repository, PostgresPatientRepository)
        else repository.create_review(patient_id)
    )
    # Transport completeness is not longitudinal record coverage. Ordinary EHR
    # requests therefore retain insufficient-data semantics for missing events.
    request = evaluation_context(context)
    findings = ClinicalReviewEngine(repository).evaluate(review.review_id, request)
    return {"cards": cards(findings)}
