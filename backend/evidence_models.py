"""Typed identities shared by retrieval, generation, and citation rendering."""

import re
from enum import StrEnum
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from backend.knowledge_models import Lifecycle, SourceType, RetrievalPolicy


class EvidenceUnitType(StrEnum):
    RECOMMENDATION = "recommendation"
    CONTEXT_CHUNK = "context_chunk"
    REFERENCE_CHUNK = "reference_chunk"


class QueryIntent(StrEnum):
    CLINICAL_GUIDANCE = "clinical_guidance"
    REFERENCE_EXPLANATION = "reference_explanation"
    HISTORICAL = "historical"


class AnswerStatus(StrEnum):
    GROUNDED = "grounded"
    ABSTAINED = "abstained"
    CONFLICT = "conflict"


class SupportStatus(StrEnum):
    SUPPORTED = "supported"
    UNCERTAIN = "uncertain"
    UNSUPPORTED = "unsupported"


class EvidenceUnit(BaseModel):
    evidence_unit_id: str = Field(min_length=1)
    unit_type: EvidenceUnitType
    chunk_id: str | None = None
    source_chunk_ids: list[str] = Field(default_factory=list)
    recommendation_id: str | None = None
    document_id: str = Field(min_length=1)
    version_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    section: str | None = None
    page_start: int | None = Field(default=None, ge=1)
    page_end: int | None = Field(default=None, ge=1)
    source_type: SourceType
    jurisdiction: str = Field(min_length=1)
    lifecycle_status: Lifecycle
    canonical_source_url: str | None = None
    publisher: str = Field(min_length=1)
    canonical_title: str = Field(min_length=1)
    published_at: str | None = None
    updated_at: str | None = None
    extractor_version: str | None = None

    @field_validator("evidence_unit_id", "document_id", "version_id")
    @classmethod
    def valid_identity(cls, value: str) -> str:
        if re.search(r"\s", value):
            raise ValueError("Evidence identity cannot contain whitespace")
        return value

    @model_validator(mode="after")
    def validate_identity(self) -> "EvidenceUnit":
        if self.page_start is not None and self.page_end is not None and self.page_end < self.page_start:
            raise ValueError("Invalid page range")
        if self.unit_type is EvidenceUnitType.RECOMMENDATION:
            if (
                not self.recommendation_id
                or not re.fullmatch(r"\d{1,2}\.\d{1,2}\.\d{1,3}", self.recommendation_id)
                or not self.extractor_version
                or not self.source_chunk_ids
                or not self.evidence_unit_id.endswith(f":rec:{self.recommendation_id}")
            ):
                raise ValueError("Recommendation requires structural provenance")
        elif self.recommendation_id is not None:
            raise ValueError("Only recommendation units may carry recommendation IDs")
        if self.unit_type is EvidenceUnitType.REFERENCE_CHUNK and self.source_type not in (
            SourceType.TEXTBOOK,
            SourceType.PATIENT_EDUCATION,
        ):
            raise ValueError("Reference unit requires reference source")
        return self


class RetrievalRequest(BaseModel):
    query: str = Field(min_length=3, max_length=2000)
    intent: QueryIntent = QueryIntent.CLINICAL_GUIDANCE
    lifecycle_policy: RetrievalPolicy | None = None
    jurisdiction: str | None = None
    source_types: list[SourceType] = Field(default_factory=list)
    document_ids: list[str] = Field(default_factory=list)
    version_ids: list[str] = Field(default_factory=list)
    recommendation_ids: list[str] = Field(default_factory=list)
    unit_types: list[EvidenceUnitType] = Field(default_factory=list)
    top_k: int = Field(default=5, ge=1, le=10)

    @property
    def policy(self) -> RetrievalPolicy:
        if self.lifecycle_policy is not None:
            return self.lifecycle_policy
        return {
            QueryIntent.CLINICAL_GUIDANCE: RetrievalPolicy.CURRENT_CLINICAL,
            QueryIntent.REFERENCE_EXPLANATION: RetrievalPolicy.REFERENCE,
            QueryIntent.HISTORICAL: RetrievalPolicy.HISTORICAL_ONLY,
        }[self.intent]


class EvidenceQueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=3, max_length=2000)
    intent: QueryIntent = QueryIntent.CLINICAL_GUIDANCE
    jurisdiction: str | None = None
    source_types: list[SourceType] = Field(default_factory=list)
    document_ids: list[str] = Field(default_factory=list)
    version_ids: list[str] = Field(default_factory=list)
    recommendation_ids: list[str] = Field(default_factory=list)
    top_k: int = Field(default=5, ge=1, le=10)

    def retrieval_request(self) -> RetrievalRequest:
        return RetrievalRequest(**self.model_dump())


class EvidenceCandidate(BaseModel):
    evidence_unit_id: str
    retriever: str
    rank: int = Field(ge=1)
    raw_score: float


class RankedEvidence(BaseModel):
    unit: EvidenceUnit
    dense_rank: int | None = None
    bm25_rank: int | None = None
    cosine_similarity: float | None = None
    bm25_score: float | None = None
    rrf_score: float
    reranker_score: float | None = None


class RetrievalDiagnostics(BaseModel):
    candidate_count: int = 0
    dense_count: int = 0
    bm25_count: int = 0
    reranker_used: bool = False
    acceptance_reason: str = ""
    top_cosine_similarity: float | None = None
    latency_ms: dict[str, float] = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    request: RetrievalRequest
    accepted: bool
    evidence: list[RankedEvidence]
    diagnostics: RetrievalDiagnostics


class AnswerClaim(BaseModel):
    claim_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    evidence_ids: list[str] = Field(min_length=1)
    support_status: SupportStatus = SupportStatus.UNCERTAIN
    verification_passages: dict[str, str] = Field(default_factory=dict)


class EvidenceConflict(BaseModel):
    conflict_id: str
    topic: str
    evidence_ids: list[str] = Field(min_length=2)
    jurisdictions: list[str]
    description: str


class EvidenceAnswer(BaseModel):
    status: AnswerStatus
    answer_text: str
    claims: list[AnswerClaim] = Field(default_factory=list)
    conflicts: list[EvidenceConflict] = Field(default_factory=list)
    evidence: list[EvidenceUnit] = Field(default_factory=list)
    retrieval: RetrievalDiagnostics = Field(default_factory=RetrievalDiagnostics)
    failure_reason: str | None = None
    rejected_claims: list[AnswerClaim] = Field(default_factory=list, exclude=True)


class EvidenceReference(BaseModel):
    evidence_unit_id: str
    unit_type: EvidenceUnitType
    document_id: str
    version_id: str
    recommendation_id: str | None = None
    publisher: str
    canonical_title: str
    canonical_source_url: str | None = None
    jurisdiction: str
    lifecycle_status: Lifecycle
    published_at: str | None = None
    updated_at: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    section: str | None = None
    supporting_excerpt: str


class EvidenceQueryResponse(BaseModel):
    status: AnswerStatus
    answer: str
    answer_text: str
    claims: list[AnswerClaim]
    conflicts: list[EvidenceConflict]
    evidence: list[EvidenceReference]
    retrieval: RetrievalDiagnostics
    failure_reason: str | None = None
