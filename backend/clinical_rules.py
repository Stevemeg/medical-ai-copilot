"""Offline, coded annual follow-up checks with governed evidence bindings."""

from __future__ import annotations

import logging
import re
from datetime import date
from typing import Protocol, cast

from backend.clinical_models import (
    FindingStatus,
    RuleDefinition,
    RuleEvaluationContext,
    RuleEvidenceReference,
    RuleResult,
    RuleStatus,
    ResourceType,
)
from backend.knowledge_models import IngestionStatus, Lifecycle, SourceType
from backend.patient_models import Concept, PatientContext
from backend.rule_evidence import (
    EvidenceBasis,
    RecommendationEvidenceError,
    RecommendationEvidenceRegistry,
    VerifiedRecommendationEvidence,
    validate_current_recommendation,
)
from backend.source_registry import SourceRegistry

SYNTHETIC_SYSTEM = "urn:medical-ai-copilot:synthetic-clinical-event"
ICD10_SYSTEM = "http://hl7.org/fhir/sid/icd-10"
HTN_CODES = {(ICD10_SYSTEM, "I10")}
T2D_CODES = {(ICD10_SYSTEM, "E11.9"), (ICD10_SYSTEM, "E11")}
HTN_REVIEW_CODE = (SYNTHETIC_SYSTEM, "HTN-ANNUAL-REVIEW")
FOOT_ASSESSMENT_CODE = (SYNTHETIC_SYSTEM, "DIABETIC-FOOT-RISK-ASSESSMENT")
LOG = logging.getLogger(__name__)


class RuleEvidenceError(ValueError):
    pass


def annual_window_start(as_of: date) -> date:
    """One calendar year earlier, with 29 February clamped to 28 February."""
    try:
        return as_of.replace(year=as_of.year - 1)
    except ValueError:
        return as_of.replace(year=as_of.year - 1, day=28)


def _has_code(concept: Concept, accepted: set[tuple[str, str]] | tuple[str, str]) -> bool:
    codes = {accepted} if isinstance(accepted, tuple) else accepted
    return any(concept.has_code(system, code) for system, code in codes)


def _condition_state(context: PatientContext, codes: set[tuple[str, str]]) -> str:
    active = [c for c in context.conditions if c.clinical_status == "active"]
    if any(_has_code(c.code, codes) for c in active):
        return "present"
    if any(not c.code.coding or any(not x.system or not x.code for x in c.code.coding) for c in active):
        return "unknown"
    family = "E11" if any(code.startswith("E11") for _, code in codes) else "I1"
    if any(
        coding.system == ICD10_SYSTEM and coding.code and coding.code.startswith(family)
        for condition in active
        for coding in condition.code.coding
    ):
        return "unknown"
    return "absent"


def _adult_state(context: PatientContext, as_of: date) -> str:
    if not context.patient.birth_date:
        return "unknown"
    born = date.fromisoformat(context.patient.birth_date)
    return "adult" if (as_of.year - born.year - ((as_of.month, as_of.day) < (born.month, born.day))) >= 18 else "minor"


class ClinicalRule(Protocol):
    definition: RuleDefinition

    def evaluate(self, context: RuleEvaluationContext) -> RuleResult: ...


class AnnualEventRule:
    def __init__(
        self,
        definition: RuleDefinition,
        condition_codes: set[tuple[str, str]],
        event_code: tuple[str, str],
        resource_type: ResourceType,
    ):
        self.definition = definition
        self.condition_codes = condition_codes
        self.event_code = event_code
        self.resource_type = resource_type

    def evaluate(self, context: RuleEvaluationContext) -> RuleResult:
        observed: dict[str, str | int | float | None] = {
            "evaluation_date": context.as_of.isoformat(),
            "review_interval_months": 12,
        }
        condition_state = _condition_state(context.patient_context, self.condition_codes)
        if condition_state == "absent":
            return RuleResult(
                status=FindingStatus.NOT_APPLICABLE,
                rationale="No supported coded active condition was supplied.",
                observed_values=observed,
            )
        if condition_state == "unknown":
            return RuleResult(
                status=FindingStatus.INSUFFICIENT_DATA,
                rationale="Active condition coding is missing or unsupported for this check.",
                observed_values=observed,
                missing_data=("condition_coding",),
            )
        adult = _adult_state(context.patient_context, context.as_of)
        if adult == "minor":
            return RuleResult(
                status=FindingStatus.NOT_APPLICABLE,
                rationale="This annual adult check does not apply to a minor.",
                observed_values=observed,
            )
        if adult == "unknown":
            return RuleResult(
                status=FindingStatus.INSUFFICIENT_DATA,
                rationale="Birth date is needed to establish adult applicability.",
                observed_values=observed,
                missing_data=("patient.birth_date",),
            )

        start = annual_window_start(context.as_of)
        observed["window_start"] = start.isoformat()
        if self.resource_type == "Encounter":
            events = [(e.type, e.start, e.status) for e in context.patient_context.encounters]
            event_label = "hypertension care review"
        else:
            events = [(p.code, p.performed, p.status) for p in context.patient_context.procedures]
            event_label = "diabetic foot-risk assessment"
        qualified = [
            (when, status) for code, when, status in events if code is not None and _has_code(code, self.event_code)
        ]
        dated = sorted(
            (
                date.fromisoformat(when[:10])
                for when, status in qualified
                if status in ("finished", "completed") and when
            ),
            reverse=True,
        )
        last = dated[0] if dated else None
        observed["last_event_date"] = last.isoformat() if last else None
        if not context.record_coverage or not context.record_coverage.covers(self.resource_type, start, context.as_of):
            return RuleResult(
                status=FindingStatus.INSUFFICIENT_DATA,
                rationale=f"Complete {self.resource_type.lower()} history for the annual interval was not asserted; absence cannot establish a care gap.",
                observed_values=observed,
                missing_data=(f"record_coverage.{self.resource_type.lower()}",),
            )
        if last and start <= last <= context.as_of:
            return RuleResult(
                status=FindingStatus.SATISFIED,
                rationale=f"A coded {event_label} is documented within the annual interval.",
                observed_values=observed,
            )
        if any(when is None for when, status in qualified if status in ("finished", "completed")):
            return RuleResult(
                status=FindingStatus.INSUFFICIENT_DATA,
                rationale=f"A coded {event_label} has no usable date, so its recency cannot be determined.",
                observed_values=observed,
                missing_data=(f"{self.definition.rule_id.lower()}.event_date",),
            )
        if any(when and date.fromisoformat(when[:10]) > context.as_of for when, _ in qualified):
            return RuleResult(
                status=FindingStatus.INSUFFICIENT_DATA,
                rationale=f"A coded {event_label} is dated after the evaluation date.",
                observed_values=observed,
                missing_data=("future_event_date",),
            )
        if not qualified and any(
            code is None
            or not code.coding
            or any(c.system == SYNTHETIC_SYSTEM and c.code != self.event_code[1] for c in code.coding)
            for code, _, status in events
            if status in ("finished", "completed")
        ):
            return RuleResult(
                status=FindingStatus.INSUFFICIENT_DATA,
                rationale=f"An event has unsupported coding; whether it was a {event_label} is unknown.",
                observed_values=observed,
                missing_data=(f"{self.resource_type.lower()}_coding",),
            )
        matching_conditions = [
            condition
            for condition in context.patient_context.conditions
            if condition.clinical_status == "active" and _has_code(condition.code, self.condition_codes)
        ]
        onset_dates = [
            date.fromisoformat(value[:10])
            for condition in matching_conditions
            if (value := condition.onset or condition.recorded_date)
        ]
        if not onset_dates or min(onset_dates) > start:
            observed["condition_onset_date"] = min(onset_dates).isoformat() if onset_dates else None
            return RuleResult(
                status=FindingStatus.INSUFFICIENT_DATA,
                rationale="The supplied onset dates do not establish that an annual reassessment is due; diagnosis-time assessment is outside this check.",
                observed_values=observed,
                missing_data=("annual_reassessment_eligibility",),
            )
        return RuleResult(
            status=FindingStatus.POTENTIAL_CARE_GAP,
            rationale=f"No coded {event_label} is documented within the fully covered annual interval. Clinician review is required.",
            observed_values=observed,
        )


class RuleRegistry:
    def __init__(
        self,
        evidence_registry: SourceRegistry,
        recommendation_registry: RecommendationEvidenceRegistry | None = None,
    ):
        self.evidence_registry = evidence_registry
        self.recommendation_registry = recommendation_registry
        if self.recommendation_registry is None:
            try:
                self.recommendation_registry = RecommendationEvidenceRegistry()
            except RecommendationEvidenceError:
                LOG.warning("recommendation_evidence_registry_unavailable")
        self.rules: dict[tuple[str, str], ClinicalRule] = {}

    def evidence(self, definition: RuleDefinition) -> RuleEvidenceReference:
        doc = self.evidence_registry.documents.get(definition.source_document_id)
        if doc is None or doc.source_type is not SourceType.CLINICAL_GUIDELINE:
            raise RuleEvidenceError("evidence document is not a registered clinical guideline")
        if definition.evidence_basis is EvidenceBasis.LOCAL_CURRENT_DOCUMENT:
            version = self.evidence_registry.versions.get(definition.source_version_id or "")
            if version is None or version.document_id != doc.document_id:
                raise RuleEvidenceError("evidence document or version is unavailable")
            if version.status is not Lifecycle.CURRENT or version.ingestion_status is not IngestionStatus.INGESTED:
                raise RuleEvidenceError("evidence is not an ingested current clinical guideline")
            if not doc.canonical_source_url:
                raise RuleEvidenceError("canonical evidence URL is unavailable")
            snapshot = self._snapshot(definition) if definition.evidence_id else None
            return RuleEvidenceReference(
                evidence_id=snapshot.evidence_id if snapshot else None,
                evidence_basis=EvidenceBasis.LOCAL_CURRENT_DOCUMENT,
                document_id=doc.document_id,
                version_id=version.version_id,
                recommendation_id=definition.recommendation_id,
                canonical_source_url=snapshot.canonical_source_url if snapshot else doc.canonical_source_url,
                source_verified_on=snapshot.verified_on if snapshot else definition.source_verified_on,
                publisher=doc.publisher,
                guideline_title=doc.canonical_title,
                guideline_code=doc.guideline_code,
                jurisdiction=doc.jurisdiction,
                lifecycle_status=version.status.value,
                verification_status=snapshot.verification_status if snapshot else None,
                recommendation_sha256=snapshot.recommendation_sha256 if snapshot else None,
            )
        if definition.evidence_basis is EvidenceBasis.AUTHORITATIVE_RECOMMENDATION_SNAPSHOT:
            if definition.source_version_id is not None:
                raise RuleEvidenceError("recommendation snapshot must not claim a local document version")
            snapshot = self._snapshot(definition)
            return RuleEvidenceReference(
                evidence_id=snapshot.evidence_id,
                evidence_basis=EvidenceBasis.AUTHORITATIVE_RECOMMENDATION_SNAPSHOT,
                document_id=doc.document_id,
                recommendation_id=snapshot.recommendation_id,
                canonical_source_url=snapshot.canonical_source_url,
                source_verified_on=snapshot.verified_on,
                publisher=snapshot.publisher,
                guideline_title=doc.canonical_title,
                guideline_code=snapshot.guideline_code,
                jurisdiction=snapshot.jurisdiction,
                verification_status=snapshot.verification_status,
                recommendation_sha256=snapshot.recommendation_sha256,
            )
        raise RuleEvidenceError("unsupported evidence basis")

    def _snapshot(self, definition: RuleDefinition) -> VerifiedRecommendationEvidence:
        if self.recommendation_registry is None or definition.evidence_id is None:
            raise RuleEvidenceError("recommendation evidence registry or ID is unavailable")
        try:
            self.recommendation_registry.refresh_if_changed()
        except RecommendationEvidenceError as exc:
            raise RuleEvidenceError("recommendation evidence registry is unavailable") from exc
        snapshot = self.recommendation_registry.evidence.get(definition.evidence_id)
        if snapshot is None:
            raise RuleEvidenceError("recommendation evidence is unavailable")
        try:
            validate_current_recommendation(snapshot)
        except RecommendationEvidenceError as exc:
            raise RuleEvidenceError("recommendation evidence is not current and authoritative") from exc
        doc = self.evidence_registry.documents.get(definition.source_document_id)
        if (
            doc is None
            or snapshot.document_id != doc.document_id
            or snapshot.publisher != doc.publisher
            or snapshot.jurisdiction != doc.jurisdiction
            or snapshot.guideline_code != doc.guideline_code
            or snapshot.recommendation_id != definition.recommendation_id
            or snapshot.verified_on != definition.source_verified_on
        ):
            raise RuleEvidenceError("recommendation evidence does not match rule or document identity")
        return snapshot

    def register(self, rule: ClinicalRule) -> None:
        d = rule.definition
        if not re.fullmatch(r"[A-Z][A-Z0-9_]+", d.rule_id) or not re.fullmatch(r"\d+\.\d+\.\d+", d.rule_version):
            raise ValueError("invalid rule ID or semantic version")
        key = (d.rule_id, d.rule_version)
        if key in self.rules:
            raise ValueError("duplicate rule ID and version")
        if d.status is RuleStatus.ACTIVE:
            self.evidence(d)
        self.rules[key] = rule

    def current_rules(self) -> tuple[ClinicalRule, ...]:
        latest: dict[str, ClinicalRule] = {}
        for rule in self.rules.values():
            d = rule.definition
            if d.rule_id not in latest or tuple(map(int, d.rule_version.split("."))) > tuple(
                map(int, latest[d.rule_id].definition.rule_version.split("."))
            ):
                latest[d.rule_id] = rule
        return tuple(latest[key] for key in sorted(latest))


def default_registry(
    evidence_registry: SourceRegistry | None = None,
    recommendation_registry: RecommendationEvidenceRegistry | None = None,
) -> RuleRegistry:
    registry = RuleRegistry(evidence_registry or SourceRegistry(), recommendation_registry)
    definitions = (
        (
            "HTN_ANNUAL_CARE_REVIEW",
            "Hypertension annual care review",
            "hypertension",
            "nice-ng136",
            "nice-ng136-2026-02-26",
            "1.4.24",
            EvidenceBasis.LOCAL_CURRENT_DOCUMENT,
            "nice-ng136-rec-1.4.24",
            HTN_CODES,
            HTN_REVIEW_CODE,
            "Encounter",
        ),
        (
            "DIABETES_ANNUAL_FOOT_ASSESSMENT",
            "Diabetes annual foot-risk assessment",
            "diabetes",
            "nice-ng19",
            None,
            "1.3.3",
            EvidenceBasis.AUTHORITATIVE_RECOMMENDATION_SNAPSHOT,
            "nice-ng19-rec-1.3.3",
            T2D_CODES,
            FOOT_ASSESSMENT_CODE,
            "Procedure",
        ),
    )
    for (
        rule_id,
        title,
        domain,
        doc_id,
        version_id,
        rec_id,
        basis,
        evidence_id,
        condition_codes,
        event_code,
        resource_type,
    ) in definitions:
        definition = RuleDefinition(
            rule_id=rule_id,
            rule_version="1.0.0",
            title=title,
            domain=domain,
            status=RuleStatus.ACTIVE,
            effective_from=date(2026, 9, 24),
            effective_to=None,
            description="Checks presence of a coded annual follow-up event in a declared complete record interval.",
            required_patient_data=("Patient.birthDate", "Condition.code", resource_type),
            source_document_id=doc_id,
            source_version_id=version_id,
            recommendation_id=rec_id,
            source_verified_on=date(2026, 9, 24),
            rule_kind="annual_coded_event",
            evidence_basis=basis,
            evidence_id=evidence_id,
        )
        try:
            registry.register(
                AnnualEventRule(definition, condition_codes, event_code, cast(ResourceType, resource_type))
            )
        except RuleEvidenceError:
            # Keep the application available, but expose this rule as suppressed.
            registry.register(
                AnnualEventRule(
                    definition.model_copy(
                        update={"status": RuleStatus.DISABLED, "suppression_reason": "rule_evidence_unavailable"}
                    ),
                    condition_codes,
                    event_code,
                    cast(ResourceType, resource_type),
                )
            )
    return registry
