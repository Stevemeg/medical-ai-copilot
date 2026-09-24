"""Offline checks for recommendation provenance, drift, and rule activation."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError

from backend.clinical_models import EvaluationRequest, FindingStatus, RecordCoverage, RuleStatus
from backend.clinical_review import ClinicalReviewEngine
from backend.clinical_rules import AnnualEventRule, RuleEvidenceError, RuleRegistry, default_registry
from backend.fhir_adapter import parse_bundle
from backend.knowledge_models import Lifecycle, RetrievalPolicy
from backend.rule_evidence import (
    EvidenceBasis,
    RecommendationEvidenceError,
    RecommendationEvidenceRegistry,
    VerificationStatus,
    VerifiedRecommendationEvidence,
    extract_recommendation,
    recommendation_fingerprint,
    validate_current_recommendation,
)
from backend.patient_store import SQLitePatientRepository
from backend.source_registry import SourceRegistry
from scripts.verify_rule_evidence import verify_registry

NG19_ID = "nice-ng19-rec-1.3.3"


def foot_rule():
    return next(rule for rule in default_registry().current_rules() if rule.definition.domain == "diabetes")


def with_definition(rule, **updates):
    return AnnualEventRule(
        rule.definition.model_copy(update=updates), rule.condition_codes, rule.event_code, rule.resource_type
    )


def synthetic_html(content: str) -> str:
    return (
        '<article id="ng19-1_3_3" class="recommendation">'
        '<h5 class="recommendation__number">1.3.3</h5>'
        f'<div class="recommendation__body"><p>{content}</p><ul><li><p>Annual check.</p></li></ul></div>'
        "</article>"
    )


def registry_file(path: Path, **changes: str) -> Path:
    row = {
        "evidence_id": NG19_ID,
        "document_id": "nice-ng19",
        "publisher": "NICE",
        "guideline_code": "NG19",
        "recommendation_id": "1.3.3",
        "canonical_source_url": "https://www.nice.org.uk/guidance/ng19/chapter/recommendations",
        "jurisdiction": "UK",
        "verified_on": "2026-09-24",
        "verification_status": "verified_current",
        "recommendation_sha256": recommendation_fingerprint("Synthetic assessment. • Annual check."),
        "summary": "Synthetic purpose.",
    }
    row.update(changes)
    path.write_text(json.dumps({"schema_version": 1, "recommendations": [row]}), encoding="utf-8")
    return path


def test_official_recommendations_registered_and_current():
    registry = RecommendationEvidenceRegistry()
    assert set(registry.evidence) == {"nice-ng136-rec-1.4.24", NG19_ID}
    for entry in registry.evidence.values():
        validate_current_recommendation(entry)
        assert entry.verified_on == date(2026, 9, 24)
        assert len(entry.recommendation_sha256) == 64


def test_duplicate_and_malformed_registry_entries(tmp_path):
    path = registry_file(tmp_path / "registry.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["recommendations"].append(dict(payload["recommendations"][0]))
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RecommendationEvidenceError, match="Cannot load"):
        RecommendationEvidenceRegistry(path)
    for change in (
        {"canonical_source_url": "http://www.nice.org.uk/guidance/ng19/chapter/recommendations"},
        {"recommendation_id": ""},
        {"recommendation_id": "bad"},
        {"recommendation_sha256": "not-a-hash"},
    ):
        registry_file(path, **change)
        with pytest.raises(RecommendationEvidenceError, match="Cannot load"):
            RecommendationEvidenceRegistry(path)


@pytest.mark.parametrize(
    "changes",
    [
        {"publisher": "Mirror"},
        {"evidence_id": "other-id"},
        {"verification_method": "unchecked"},
        {"guideline_code": "BAD"},
        {"canonical_source_url": "https://nice.org.uk.evil.example/guidance/ng19/chapter/recommendations"},
        {"canonical_source_url": "https://github.com/guidance/ng19/chapter/recommendations"},
        {"canonical_source_url": "https://www.nice.org.uk/guidance/ng28/chapter/recommendations"},
        {"verification_status": VerificationStatus.UNVERIFIED},
        {"verification_status": VerificationStatus.DRIFT_DETECTED},
        {"verification_status": VerificationStatus.SUPERSEDED},
        {"verification_status": VerificationStatus.WITHDRAWN},
    ],
)
def test_untrusted_or_noncurrent_recommendation_cannot_activate(changes):
    recs = RecommendationEvidenceRegistry()
    recs.evidence[NG19_ID] = recs.evidence[NG19_ID].model_copy(update=changes)
    with pytest.raises(RuleEvidenceError):
        RuleRegistry(SourceRegistry(), recs).register(foot_rule())


def test_document_and_recommendation_identity_must_match():
    recs = RecommendationEvidenceRegistry()
    rule = foot_rule()
    for change in ({"document_id": "nice-ng136"}, {"recommendation_id": "1.3.4"}, {"verified_on": date(2025, 1, 1)}):
        recs.evidence[NG19_ID] = RecommendationEvidenceRegistry().evidence[NG19_ID].model_copy(update=change)
        with pytest.raises(RuleEvidenceError):
            RuleRegistry(SourceRegistry(), recs).register(rule)


def test_both_evidence_paths_and_stale_pdf_defense():
    registry = default_registry()
    hypertension = next(r for r in registry.current_rules() if r.definition.domain == "hypertension")
    foot = foot_rule()
    assert hypertension.definition.status is RuleStatus.ACTIVE
    assert foot.definition.status is RuleStatus.ACTIVE
    assert registry.evidence(hypertension.definition).evidence_basis is EvidenceBasis.LOCAL_CURRENT_DOCUMENT
    foot_evidence = registry.evidence(foot.definition)
    assert foot_evidence.evidence_basis is EvidenceBasis.AUTHORITATIVE_RECOMMENDATION_SNAPSHOT
    assert foot_evidence.version_id is None
    assert foot_evidence.verification_status is VerificationStatus.VERIFIED_CURRENT
    old_pdf_rule = with_definition(
        foot,
        rule_id="OLD_NG19_PDF_RULE",
        source_version_id="nice-ng19-2019-10-11",
        evidence_basis=EvidenceBasis.LOCAL_CURRENT_DOCUMENT,
        evidence_id=None,
    )
    with pytest.raises(RuleEvidenceError):
        RuleRegistry(SourceRegistry()).register(old_pdf_rule)
    ng28_rule = with_definition(
        hypertension,
        rule_id="STALE_NG28_RULE",
        source_document_id="nice-ng28",
        source_version_id="nice-ng28-2022-06-29",
        evidence_id=None,
    )
    with pytest.raises(RuleEvidenceError):
        RuleRegistry(SourceRegistry()).register(ng28_rule)
    sources = SourceRegistry()
    assert sources.versions["nice-ng19-2019-10-11"].status is Lifecycle.SUPERSEDED
    assert sources.versions["nice-ng28-2022-06-29"].status is Lifecycle.SUPERSEDED


def test_missing_evidence_fields_and_version_selection():
    foot = foot_rule()
    with pytest.raises(RuleEvidenceError):
        RuleRegistry(SourceRegistry()).register(with_definition(foot, source_version_id="nice-ng19-2019-10-11"))
    with pytest.raises(RuleEvidenceError):
        RuleRegistry(SourceRegistry()).register(with_definition(foot, evidence_id="nice-ng19-rec-9.9.9"))
    sources = SourceRegistry()
    hypertension = next(r for r in default_registry().current_rules() if r.definition.domain == "hypertension")
    doc = sources.documents["nice-ng136"]
    sources.documents["nice-ng136"] = replace(doc, canonical_source_url=None)
    with pytest.raises(RuleEvidenceError):
        RuleRegistry(sources).register(hypertension)
    registry = RuleRegistry(SourceRegistry())
    registry.register(hypertension)
    registry.register(with_definition(hypertension, rule_version="1.1.0"))
    assert registry.current_rules()[0].definition.rule_version == "1.1.0"


def test_fingerprint_normalization_and_extraction():
    first = extract_recommendation(synthetic_html("Synthetic assessment."), "NG19", "1.3.3")
    second = extract_recommendation(synthetic_html("Synthetic\n   assessment."), "NG19", "1.3.3")
    changed = extract_recommendation(synthetic_html("Different assessment."), "NG19", "1.3.3")
    assert first and second and changed
    assert recommendation_fingerprint(first) == recommendation_fingerprint(second)
    assert recommendation_fingerprint(first) != recommendation_fingerprint(changed)
    assert extract_recommendation(synthetic_html("Synthetic assessment."), "NG19", "1.3.4") is None
    assert extract_recommendation(synthetic_html("Line<br>break."), "NG19", "1.3.3") is not None


def test_online_governance_check_fail_closed_without_network_in_tests(tmp_path):
    path = registry_file(tmp_path / "registry.json")
    assert verify_registry(path, lambda _url: synthetic_html("Synthetic assessment.")) == {NG19_ID: "MATCH"}
    assert verify_registry(path, lambda _url: synthetic_html("Changed assessment.")) == {NG19_ID: "DRIFT"}
    recs = RecommendationEvidenceRegistry(path)
    assert recs.evidence[NG19_ID].verification_status is VerificationStatus.DRIFT_DETECTED
    with pytest.raises(RuleEvidenceError):
        RuleRegistry(SourceRegistry(), recs).register(foot_rule())
    assert verify_registry(path, lambda _url: synthetic_html("Synthetic assessment.")) == {NG19_ID: "MATCH"}
    assert (
        RecommendationEvidenceRegistry(path).evidence[NG19_ID].verification_status is VerificationStatus.DRIFT_DETECTED
    )

    def fail_fetch(_url: str) -> str:
        raise OSError("offline")

    assert verify_registry(path, fail_fetch)[NG19_ID].startswith("FETCH_FAILED")
    path = registry_file(tmp_path / "missing.json")
    assert verify_registry(path, lambda _url: "<html></html>") == {NG19_ID: "NOT_FOUND"}
    assert (
        RecommendationEvidenceRegistry(path).evidence[NG19_ID].verification_status is VerificationStatus.DRIFT_DETECTED
    )


def test_status_enum_and_retrieval_source_unchanged():
    row = RecommendationEvidenceRegistry().evidence[NG19_ID].model_dump(mode="json")
    row["verification_status"] = "whatever"
    with pytest.raises(ValidationError):
        VerifiedRecommendationEvidence.model_validate(row)
    sources = SourceRegistry()
    ng19 = sources.versions["nice-ng19-2019-10-11"]
    assert not sources.eligible(
        {"document_id": "nice-ng19", "version_id": ng19.version_id},
        RetrievalPolicy.CURRENT_CLINICAL,
    )


def test_default_registry_exposes_safe_suppression_reason_on_drift(tmp_path):
    recs = RecommendationEvidenceRegistry()
    recs.evidence[NG19_ID] = recs.evidence[NG19_ID].model_copy(
        update={"verification_status": VerificationStatus.DRIFT_DETECTED}
    )
    registry = default_registry(recommendation_registry=recs)
    foot = next(rule for rule in registry.current_rules() if rule.definition.domain == "diabetes")
    assert foot.definition.status is RuleStatus.DISABLED
    assert foot.definition.suppression_reason == "rule_evidence_unavailable"
    fixture = Path(__file__).resolve().parents[1] / "data" / "synthetic_fhir" / "syn_pat_001.json"
    repo = SQLitePatientRepository(tmp_path / "drift.db")
    patient_id = repo.import_context(parse_bundle(json.loads(fixture.read_text(encoding="utf-8")))[0])[0]
    review = repo.create_review(patient_id)
    request = EvaluationRequest(
        as_of=date(2026, 9, 24),
        record_coverage=RecordCoverage(
            start_date=date(2025, 9, 24),
            end_date=date(2026, 9, 24),
            complete_resource_types=("Procedure",),
        ),
    )
    finding = next(
        finding
        for finding in ClinicalReviewEngine(repo, registry).evaluate(review.review_id, request)
        if finding.domain == "diabetes"
    )
    assert finding.status is FindingStatus.SUPPRESSED
    assert finding.suppression_reason == "rule_evidence_unavailable"


def test_live_local_registry_change_suppresses_rule_without_network(tmp_path):
    path = registry_file(
        tmp_path / "registry.json",
        recommendation_sha256=RecommendationEvidenceRegistry().evidence[NG19_ID].recommendation_sha256,
    )
    recs = RecommendationEvidenceRegistry(path)
    registry = RuleRegistry(SourceRegistry(), recs)
    rule = foot_rule()
    registry.register(rule)
    assert registry.evidence(rule.definition).verification_status is VerificationStatus.VERIFIED_CURRENT
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["recommendations"][0]["verification_status"] = "drift_detected"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuleEvidenceError):
        registry.evidence(rule.definition)
    path.unlink()
    with pytest.raises(RuleEvidenceError):
        registry.evidence(rule.definition)


def test_unavailable_recommendation_registry_disables_rules_safely(monkeypatch):
    def unavailable():
        raise RecommendationEvidenceError("unavailable")

    monkeypatch.setattr("backend.clinical_rules.RecommendationEvidenceRegistry", unavailable)
    registry = default_registry()
    assert all(rule.definition.status is RuleStatus.DISABLED for rule in registry.current_rules())
    assert all(rule.definition.suppression_reason == "rule_evidence_unavailable" for rule in registry.current_rules())
