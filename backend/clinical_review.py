"""Snapshot-based deterministic evaluation and immutable finding persistence."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone

from backend.clinical_models import (
    ClinicalFinding,
    EvaluationRequest,
    FindingStatus,
    RuleEvaluationContext,
    RuleEvidenceReference,
    RuleResult,
    RuleStatus,
)
from backend.clinical_rules import RuleEvidenceError, RuleRegistry, default_registry
from backend.patient_models import ReviewStatus
from backend.patient_store import ReviewCompleted, SQLitePatientRepository

LOG = logging.getLogger(__name__)


class EvaluationConflict(ValueError):
    pass


def finding_id(review_id: str, rule_id: str, rule_version: str, context_hash: str, request: EvaluationRequest) -> str:
    payload = {
        "review_id": review_id,
        "rule_id": rule_id,
        "rule_version": rule_version,
        "context_hash": context_hash,
        "request": request.model_dump(mode="json"),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return "finding-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


class ClinicalReviewEngine:
    def __init__(self, repository: SQLitePatientRepository, registry: RuleRegistry | None = None):
        self.repository = repository
        self.registry = registry or default_registry()

    def evaluate(self, review_id: str, request: EvaluationRequest) -> tuple[ClinicalFinding, ...]:
        review = self.repository.get_review(review_id)
        if review.status is ReviewStatus.COMPLETED:
            raise ReviewCompleted(review_id)
        existing = self.repository.list_findings(review_id)
        if existing:
            if all(
                f.evaluated_as_of == request.as_of and f.record_coverage == request.record_coverage for f in existing
            ):
                return existing
            raise EvaluationConflict("Review already has findings for another evaluation context; create a new review")
        context = RuleEvaluationContext(
            patient_context=review.patient_snapshot,
            patient_context_hash=review.patient_context_hash,
            as_of=request.as_of,
            record_coverage=request.record_coverage,
        )
        findings: list[ClinicalFinding] = []
        for rule in self.registry.current_rules():
            definition = rule.definition
            evidence: tuple[RuleEvidenceReference, ...] = ()
            suppression_reason: str | None = None
            if (
                definition.status is not RuleStatus.ACTIVE
                or request.as_of < definition.effective_from
                or (definition.effective_to and request.as_of > definition.effective_to)
            ):
                result = RuleResult(
                    status=FindingStatus.SUPPRESSED,
                    rationale="Rule is inactive for this evaluation date.",
                    observed_values={"evaluation_date": request.as_of.isoformat()},
                    missing_data=("rule_availability",),
                )
                suppression_reason = definition.suppression_reason or "rule_inactive"
            else:
                try:
                    evidence = (self.registry.evidence(definition),)
                except RuleEvidenceError as exc:
                    LOG.warning("rule_suppressed rule_id=%s reason=%s", definition.rule_id, type(exc).__name__)
                    result = RuleResult(
                        status=FindingStatus.SUPPRESSED,
                        rationale="Current governed clinical evidence is unavailable for this rule.",
                        observed_values={"evaluation_date": request.as_of.isoformat()},
                        missing_data=("rule_evidence",),
                    )
                    suppression_reason = "rule_evidence_unavailable"
                else:
                    result = rule.evaluate(context)
            findings.append(
                ClinicalFinding(
                    finding_id=finding_id(
                        review_id, definition.rule_id, definition.rule_version, review.patient_context_hash, request
                    ),
                    review_id=review_id,
                    patient_context_hash=review.patient_context_hash,
                    rule_id=definition.rule_id,
                    rule_version=definition.rule_version,
                    domain=definition.domain,
                    title=definition.title,
                    status=result.status,
                    evaluated_as_of=request.as_of,
                    record_coverage=request.record_coverage,
                    rationale=result.rationale,
                    observed_values=result.observed_values,
                    missing_data=result.missing_data,
                    evidence_refs=evidence,
                    suppression_reason=suppression_reason,
                    created_at=datetime.now(timezone.utc),
                )
            )
        saved = self.repository.save_findings(review_id, tuple(findings))
        LOG.info("review_evaluated review_id=%s finding_count=%d", review_id, len(saved))
        return saved
