"""Normalized, immutable patient and review records for the supported FHIR subset."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class Record(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Coding(Record):
    system: str | None = None
    code: str | None = None
    display: str | None = None


class Concept(Record):
    text: str | None = None
    coding: tuple[Coding, ...] = ()

    @property
    def display(self) -> str:
        return self.text or next(
            (c.display or c.code or "Unknown" for c in self.coding if c.display or c.code), "Unknown"
        )

    def has_code(self, system: str, code: str) -> bool:
        return any(c.system == system and c.code == code for c in self.coding)


class Patient(Record):
    source_patient_id: str
    synthetic_label: str
    birth_date: str | None = None
    gender: str | None = None


class Condition(Record):
    condition_id: str
    code: Concept
    clinical_status: str | None = None
    onset: str | None = None
    recorded_date: str | None = None


class Quantity(Record):
    value: float
    unit: str | None = None
    system: str | None = None
    code: str | None = None


class ObservationComponent(Record):
    code: Concept
    value: Quantity


class Observation(Record):
    observation_id: str
    code: Concept
    status: str
    effective: str | None = None
    value: Quantity | None = None
    components: tuple[ObservationComponent, ...] = ()


class Medication(Record):
    medication_id: str
    code: Concept
    status: str
    intent: str
    authored_on: str | None = None
    dosage_text: str | None = None


class Allergy(Record):
    allergy_id: str
    code: Concept
    clinical_status: str | None = None
    verification_status: str | None = None
    category: tuple[str, ...] = ()
    criticality: str | None = None
    recorded_date: str | None = None
    reactions: tuple[str, ...] = ()


class Encounter(Record):
    encounter_id: str
    status: str
    class_code: str | None = None
    type: Concept | None = None
    start: str | None = None
    end: str | None = None


class Procedure(Record):
    procedure_id: str
    code: Concept
    status: str
    performed: str | None = None


class PatientSourceMetadata(Record):
    format: str = "FHIR R4-compatible supported subset"
    synthetic: bool = True


class PatientContext(Record):
    patient: Patient
    conditions: tuple[Condition, ...] = ()
    observations: tuple[Observation, ...] = ()
    medications: tuple[Medication, ...] = ()
    allergies: tuple[Allergy, ...] = ()
    encounters: tuple[Encounter, ...] = ()
    procedures: tuple[Procedure, ...] = ()
    source_metadata: PatientSourceMetadata = Field(default_factory=PatientSourceMetadata)


class TimelineEvent(Record):
    event_id: str
    event_type: str
    timestamp: str | None
    title: str
    details: dict[str, str] = Field(default_factory=dict)
    source_resource_id: str


class DataAvailability(Record):
    available_data_types: tuple[str, ...]
    missing_data_types: tuple[str, ...]
    last_observation_dates: dict[str, str]


class ReviewStatus(StrEnum):
    READY_FOR_REVIEW = "ready_for_review"
    COMPLETED = "completed"


class ClinicalReview(Record):
    review_id: str
    patient_id: str
    patient_context_hash: str
    created_at: str
    status: ReviewStatus
    patient_snapshot: PatientContext
    timeline: tuple[TimelineEvent, ...]
    data_availability: DataAvailability
    findings: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()
    clinician_actions: tuple[str, ...] = ()
    completed_at: str | None = None
