"""Typed, immutable contracts for deterministic clinical review."""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from backend.patient_models import PatientContext


class ClinicalRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class RuleStatus(StrEnum):
    ACTIVE = "active"
    DISABLED = "disabled"
    RETIRED = "retired"


class FindingStatus(StrEnum):
    SATISFIED = "satisfied"
    POTENTIAL_CARE_GAP = "potential_care_gap"
    INSUFFICIENT_DATA = "insufficient_data"
    NOT_APPLICABLE = "not_applicable"
    SUPPRESSED = "suppressed"


class ActionType(StrEnum):
    ACCEPT = "accept"
    DISMISS = "dismiss"
    ALREADY_ADDRESSED = "already_addressed"
    INCORRECT_EVIDENCE = "incorrect_evidence"
    NOT_CLINICALLY_RELEVANT = "not_clinically_relevant"


class RuleEvidenceReference(ClinicalRecord):
    document_id: str
    version_id: str
    recommendation_id: str
    canonical_source_url: str
    source_verified_on: date
    publisher: str = "NICE"
    guideline_title: str
    lifecycle_status: str = "current"


class RuleDefinition(ClinicalRecord):
    rule_id: str
    rule_version: str
    title: str
    domain: str
    status: RuleStatus
    effective_from: date
    effective_to: date | None = None
    description: str
    required_patient_data: tuple[str, ...]
    source_document_id: str
    source_version_id: str
    recommendation_id: str
    source_verified_on: date
    rule_kind: str


ResourceType = Literal["Encounter", "Procedure", "Condition", "Observation"]


class RecordCoverage(ClinicalRecord):
    """Separate demo assertion about completeness, not FHIR metadata."""

    start_date: date
    end_date: date
    complete_resource_types: tuple[ResourceType, ...]

    @model_validator(mode="after")
    def valid_range(self) -> RecordCoverage:
        if self.end_date < self.start_date:
            raise ValueError("coverage end precedes start")
        if len(set(self.complete_resource_types)) != len(self.complete_resource_types):
            raise ValueError("duplicate completeness declaration")
        return self

    def covers(self, resource_type: ResourceType, start: date, end: date) -> bool:
        return resource_type in self.complete_resource_types and self.start_date <= start and self.end_date >= end


class EvaluationRequest(ClinicalRecord):
    as_of: date
    record_coverage: RecordCoverage | None = None

    @model_validator(mode="after")
    def valid_time(self) -> EvaluationRequest:
        if self.as_of < date(1900, 1, 1) or self.as_of > date(2100, 12, 31):
            raise ValueError("evaluation date outside supported range")
        if self.record_coverage and self.record_coverage.end_date > self.as_of:
            raise ValueError("coverage cannot extend beyond evaluation date")
        return self


class RuleEvaluationContext(ClinicalRecord):
    patient_context: PatientContext
    patient_context_hash: str
    as_of: date
    record_coverage: RecordCoverage | None = None


class RuleResult(ClinicalRecord):
    status: FindingStatus
    rationale: str
    observed_values: dict[str, str | int | float | None]
    missing_data: tuple[str, ...] = ()


class ClinicalFinding(ClinicalRecord):
    finding_id: str
    review_id: str
    patient_context_hash: str
    rule_id: str
    rule_version: str
    domain: str
    title: str
    status: FindingStatus
    evaluated_as_of: date
    record_coverage: RecordCoverage | None
    rationale: str
    observed_values: dict[str, str | int | float | None]
    missing_data: tuple[str, ...]
    evidence_refs: tuple[RuleEvidenceReference, ...]
    created_at: datetime


class FindingAction(ClinicalRecord):
    action_id: str
    finding_id: str
    action_type: ActionType
    note: str | None = None
    created_at: datetime
    actor: str | None = None

    @field_validator("note")
    @classmethod
    def bounded_note(cls, value: str | None) -> str | None:
        if value is not None and len(value) > 2000:
            raise ValueError("note exceeds 2000 characters")
        return value


class ActionRequest(ClinicalRecord):
    action_type: ActionType
    note: str | None = None
