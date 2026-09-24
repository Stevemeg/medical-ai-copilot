"""Validated identities and provenance for locally indexed evidence."""

from dataclasses import dataclass
from enum import StrEnum


class SourceType(StrEnum):
    CLINICAL_GUIDELINE = "clinical_guideline"
    SYSTEMATIC_REVIEW = "systematic_review"
    PUBLIC_HEALTH_REPORT = "public_health_report"
    ORIGINAL_RESEARCH = "original_research"
    TEXTBOOK = "textbook"
    PATIENT_EDUCATION = "patient_education"
    OTHER = "other"


class Lifecycle(StrEnum):
    CURRENT = "current"
    SUPERSEDED = "superseded"
    HISTORICAL = "historical"
    WITHDRAWN = "withdrawn"
    UNKNOWN = "unknown"


class IngestionStatus(StrEnum):
    PENDING = "pending"
    VALIDATED = "validated"
    INGESTED = "ingested"
    FAILED = "failed"
    QUARANTINED = "quarantined"


class RetrievalPolicy(StrEnum):
    CURRENT_CLINICAL = "current_clinical"
    HISTORICAL_ALLOWED = "historical_allowed"
    HISTORICAL_ONLY = "historical_only"
    REFERENCE = "reference"


@dataclass(frozen=True)
class KnowledgeDocument:
    document_id: str
    canonical_title: str
    publisher: str
    source_type: SourceType
    jurisdiction: str
    guideline_code: str | None
    canonical_source_url: str | None
    license: str | None


@dataclass(frozen=True)
class KnowledgeDocumentVersion:
    version_id: str
    document_id: str
    version_label: str | None
    published_at: str | None
    updated_at: str | None
    retrieved_at: str | None
    valid_from: str | None
    valid_until: str | None
    status: Lifecycle
    supersedes_version_id: str | None
    superseded_by_version_id: str | None
    sha256: str
    parser_version: str
    ingestion_status: IngestionStatus
    source_path: str
    legacy_source: str


@dataclass(frozen=True)
class EvidenceChunk:
    chunk_id: str
    document_id: str
    version_id: str
    text: str
    page_start: int | None
    page_end: int | None
    section: str | None
    recommendation_id: str | None
    content_sha256: str
    source: str
