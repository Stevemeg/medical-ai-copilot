"""Deterministic annual review rules, evidence governance, and human dispositions."""

from __future__ import annotations

import json
import socket
import sqlite3
from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.clinical_models import (
    ActionType,
    EvaluationRequest,
    FindingStatus,
    RecordCoverage,
    RuleEvaluationContext,
    RuleStatus,
)
from backend.clinical_review import ClinicalReviewEngine, EvaluationConflict
from backend.clinical_rules import AnnualEventRule, RuleEvidenceError, RuleRegistry, default_registry
from backend.fhir_adapter import parse_bundle
from backend.knowledge_models import IngestionStatus, Lifecycle, SourceType
from backend.patient_api import get_repository
from backend.patient_context import context_hash, latest_observation, observations_with_unknown_date
from backend.patient_store import ReviewCompleted, SQLitePatientRepository
from backend.source_registry import SourceRegistry

FIXTURES = Path(__file__).resolve().parents[1] / "data" / "synthetic_fhir"
MANIFEST = json.loads((FIXTURES / "manifest.json").read_text(encoding="utf-8"))["fixtures"]


def raw(number: int) -> dict:
    return json.loads((FIXTURES / f"syn_pat_{number:03d}.json").read_text(encoding="utf-8"))


def setup_review(tmp_path: Path, number: int) -> tuple[SQLitePatientRepository, str]:
    repo = SQLitePatientRepository(tmp_path / "clinical.db")
    patient_id = repo.import_context(parse_bundle(raw(number))[0])[0]
    return repo, repo.create_review(patient_id).review_id


def request(number: int, *, coverage: bool = True) -> EvaluationRequest:
    row = MANIFEST[f"syn_pat_{number:03d}"]
    return EvaluationRequest(
        as_of=date.fromisoformat(row["evaluation_as_of"]),
        record_coverage=RecordCoverage.model_validate(row["record_coverage"])
        if coverage and row["record_coverage"]
        else None,
    )


def candidate_foot_result(bundle: dict, evaluation: EvaluationRequest):
    """Exercise inactive Rule B logic without registering stale evidence as active."""
    context = parse_bundle(bundle)[0]
    rule = next(r for r in default_registry().current_rules() if r.definition.domain == "diabetes")
    assert rule.definition.status is RuleStatus.DISABLED
    return rule.evaluate(
        RuleEvaluationContext(
            patient_context=context,
            patient_context_hash=context_hash(context),
            as_of=evaluation.as_of,
            record_coverage=evaluation.record_coverage,
        )
    )


@pytest.mark.parametrize(
    "number,rule_id,expected",
    [
        (1, "DIABETES_ANNUAL_FOOT_ASSESSMENT", FindingStatus.SUPPRESSED),
        (2, "HTN_ANNUAL_CARE_REVIEW", FindingStatus.POTENTIAL_CARE_GAP),
        (3, "HTN_ANNUAL_CARE_REVIEW", FindingStatus.SATISFIED),
        (3, "DIABETES_ANNUAL_FOOT_ASSESSMENT", FindingStatus.SUPPRESSED),
        (4, "HTN_ANNUAL_CARE_REVIEW", FindingStatus.INSUFFICIENT_DATA),
        (4, "DIABETES_ANNUAL_FOOT_ASSESSMENT", FindingStatus.SUPPRESSED),
        (5, "HTN_ANNUAL_CARE_REVIEW", FindingStatus.NOT_APPLICABLE),
        (5, "DIABETES_ANNUAL_FOOT_ASSESSMENT", FindingStatus.SUPPRESSED),
    ],
)
def test_manifest_scenarios(tmp_path, number, rule_id, expected):
    repo, review_id = setup_review(tmp_path, number)
    findings = ClinicalReviewEngine(repo).evaluate(review_id, request(number))
    finding = next(f for f in findings if f.rule_id == rule_id)
    assert finding.status is expected
    assert finding.patient_context_hash == repo.get_review(review_id).patient_context_hash
    assert finding.rule_version == "1.0.0"


@pytest.mark.parametrize(
    "number,expected",
    [
        (1, FindingStatus.SATISFIED),
        (3, FindingStatus.POTENTIAL_CARE_GAP),
        (4, FindingStatus.INSUFFICIENT_DATA),
        (5, FindingStatus.NOT_APPLICABLE),
    ],
)
def test_inactive_foot_rule_logic_only(number, expected):
    assert candidate_foot_result(raw(number), request(number)).status is expected


def test_unknown_and_partial_coverage_never_create_gap(tmp_path):
    repo, review_id = setup_review(tmp_path, 2)
    finding = next(
        f
        for f in ClinicalReviewEngine(repo).evaluate(review_id, request(2, coverage=False))
        if f.domain == "hypertension"
    )
    assert finding.status is FindingStatus.INSUFFICIENT_DATA
    assert "record_coverage.encounter" in finding.missing_data
    repo2, review2 = setup_review(tmp_path / "other", 2)
    partial = EvaluationRequest(
        as_of=date(2026, 9, 24),
        record_coverage=RecordCoverage(
            start_date=date(2026, 1, 1), end_date=date(2026, 9, 24), complete_resource_types=("Encounter",)
        ),
    )
    assert (
        next(f for f in ClinicalReviewEngine(repo2).evaluate(review2, partial) if f.domain == "hypertension").status
        is FindingStatus.INSUFFICIENT_DATA
    )
    repo3, review3 = setup_review(tmp_path / "wrong_type", 3)
    wrong_type = EvaluationRequest(
        as_of=date(2026, 9, 24),
        record_coverage=RecordCoverage(
            start_date=date(2025, 9, 24), end_date=date(2026, 9, 24), complete_resource_types=("Encounter",)
        ),
    )
    assert candidate_foot_result(raw(3), wrong_type).status is FindingStatus.INSUFFICIENT_DATA
    assert (
        next(f for f in ClinicalReviewEngine(repo3).evaluate(review3, wrong_type) if f.domain == "diabetes").status
        is FindingStatus.SUPPRESSED
    )


@pytest.mark.parametrize(
    "bad",
    [
        {"start_date": "2026-09-25", "end_date": "2026-09-24", "complete_resource_types": ["Encounter"]},
        {"start_date": "2025-09-24", "end_date": "2026-09-24", "complete_resource_types": ["Encounter", "Encounter"]},
        {"start_date": "2025-09-24", "end_date": "2026-09-24", "complete_resource_types": ["Unknown"]},
    ],
)
def test_coverage_validation(bad):
    with pytest.raises(ValueError):
        RecordCoverage.model_validate(bad)
    with pytest.raises(ValueError):
        EvaluationRequest(
            as_of=date(2026, 9, 24),
            record_coverage=RecordCoverage(
                start_date=date(2025, 1, 1), end_date=date(2026, 9, 25), complete_resource_types=("Encounter",)
            ),
        )


def test_latest_observation_ignores_undated_and_ties():
    bundle = raw(1)
    observation = next(e["resource"] for e in bundle["entry"] if e["resource"]["resourceType"] == "Observation")
    observation.pop("effectiveDateTime")
    context = parse_bundle(bundle)[0]
    assert latest_observation(context, "http://loinc.org", "4548-4") is None
    assert len(observations_with_unknown_date(context, "http://loinc.org", "4548-4")) == 1
    second = json.loads(json.dumps(observation))
    second["id"] = "obs-a2"
    second["effectiveDateTime"] = "2025-08-12"
    bundle["entry"].append({"resource": second})
    assert latest_observation(parse_bundle(bundle)[0], "http://loinc.org", "4548-4").observation_id == "obs-a2"
    third = json.loads(json.dumps(second))
    third["id"] = "obs-a3"
    bundle["entry"].append({"resource": third})
    assert latest_observation(parse_bundle(bundle)[0], "http://loinc.org", "4548-4").observation_id == "obs-a3"


def test_registry_rejects_stale_ng28_and_evidence_variants():
    sources = SourceRegistry()
    registry = default_registry(sources)
    rule = next(r for r in registry.current_rules() if r.definition.domain == "hypertension")
    stale = AnnualEventRule(
        rule.definition.model_copy(
            update={
                "rule_id": "STALE_NG28_RULE",
                "source_document_id": "nice-ng28",
                "source_version_id": "nice-ng28-2022-06-29",
            }
        ),
        rule.condition_codes,
        rule.event_code,
        rule.resource_type,
    )
    with pytest.raises(RuleEvidenceError):
        registry.register(stale)
    with pytest.raises(ValueError, match="duplicate"):
        registry.register(rule)
    for status in (Lifecycle.SUPERSEDED, Lifecycle.HISTORICAL, Lifecycle.WITHDRAWN, Lifecycle.UNKNOWN):
        altered = SourceRegistry()
        altered.versions[rule.definition.source_version_id] = replace(
            altered.versions[rule.definition.source_version_id], status=status
        )
        with pytest.raises(RuleEvidenceError):
            RuleRegistry(altered).register(rule)
    altered = SourceRegistry()
    altered.versions[rule.definition.source_version_id] = replace(
        altered.versions[rule.definition.source_version_id], ingestion_status=IngestionStatus.PENDING
    )
    with pytest.raises(RuleEvidenceError):
        RuleRegistry(altered).register(rule)
    altered = SourceRegistry()
    altered.documents[rule.definition.source_document_id] = replace(
        altered.documents[rule.definition.source_document_id], source_type=SourceType.TEXTBOOK
    )
    with pytest.raises(RuleEvidenceError):
        RuleRegistry(altered).register(rule)
    for change in ({"source_document_id": "missing"}, {"source_version_id": "missing"}):
        with pytest.raises(RuleEvidenceError):
            RuleRegistry(SourceRegistry()).register(
                AnnualEventRule(
                    rule.definition.model_copy(update=change), rule.condition_codes, rule.event_code, rule.resource_type
                )
            )
    with pytest.raises(ValueError, match="version"):
        RuleRegistry(SourceRegistry()).register(
            AnnualEventRule(
                rule.definition.model_copy(update={"rule_version": "bad"}),
                rule.condition_codes,
                rule.event_code,
                rule.resource_type,
            )
        )


def test_disabled_and_drift_suppress(tmp_path):
    sources = SourceRegistry()
    registry = default_registry(sources)
    rule = next(r for r in registry.current_rules() if r.definition.domain == "hypertension")
    disabled_registry = RuleRegistry(sources)
    disabled_registry.register(
        AnnualEventRule(
            rule.definition.model_copy(update={"status": RuleStatus.DISABLED}),
            rule.condition_codes,
            rule.event_code,
            rule.resource_type,
        )
    )
    repo, review_id = setup_review(tmp_path, 2)
    assert (
        ClinicalReviewEngine(repo, disabled_registry).evaluate(review_id, request(2))[0].status
        is FindingStatus.SUPPRESSED
    )
    sources2 = SourceRegistry()
    drift_registry = default_registry(sources2)
    sources2.versions[rule.definition.source_version_id] = replace(
        sources2.versions[rule.definition.source_version_id], status=Lifecycle.SUPERSEDED
    )
    repo2, review2 = setup_review(tmp_path / "drift", 2)
    finding = next(
        f
        for f in ClinicalReviewEngine(repo2, drift_registry).evaluate(review2, request(2))
        if f.domain == "hypertension"
    )
    assert finding.status is FindingStatus.SUPPRESSED
    assert not finding.evidence_refs


def test_idempotent_snapshot_version_and_actions(tmp_path):
    repo, review_id = setup_review(tmp_path, 3)
    engine = ClinicalReviewEngine(repo)
    first = engine.evaluate(review_id, request(3))
    assert first == engine.evaluate(review_id, request(3))
    with pytest.raises(EvaluationConflict):
        engine.evaluate(review_id, request(3, coverage=False))
    finding = first[0]
    for action_type in ActionType:
        repo.add_action(finding.finding_id, action_type)
    assert len({a.action_id for a in repo.list_actions(finding.finding_id)}) == 5
    assert repo.get_finding(finding.finding_id) == finding
    changed = raw(3)
    changed_event = next(e["resource"] for e in changed["entry"] if e["resource"].get("id") == "enc-c3")
    changed_event["period"] = {"start": "2024-03-10T11:00:00Z", "end": "2024-03-10T11:25:00Z"}
    patient_id = repo.get_review(review_id).patient_id
    repo.import_context(parse_bundle(changed)[0])
    assert repo.get_review(review_id).patient_id == patient_id
    assert repo.list_findings(review_id) == first
    new_review = repo.create_review(patient_id)
    assert new_review.patient_context_hash != finding.patient_context_hash
    assert ClinicalReviewEngine(repo).evaluate(new_review.review_id, request(3)) != first
    rule = next(r for r in default_registry().current_rules() if r.definition.domain == "hypertension")
    upgraded = RuleRegistry(SourceRegistry())
    upgraded.register(
        AnnualEventRule(
            rule.definition.model_copy(update={"rule_version": "1.1.0"}),
            rule.condition_codes,
            rule.event_code,
            rule.resource_type,
        )
    )
    assert repo.get_finding(finding.finding_id).rule_version == "1.0.0"
    version_review = repo.create_review(patient_id)
    assert (
        ClinicalReviewEngine(repo, upgraded).evaluate(version_review.review_id, request(3))[0].rule_version == "1.1.0"
    )
    repo.complete_review(review_id)
    with pytest.raises(ReviewCompleted):
        engine.evaluate(review_id, request(3))


def test_api_findings_actions_and_errors(tmp_path):
    from api_server import app

    repo = SQLitePatientRepository(tmp_path / "api.db")
    app.dependency_overrides[get_repository] = lambda: repo
    try:
        with TestClient(app) as client:
            patient_id = client.post("/v1/patients/import", json=raw(2)).json()["patient_id"]
            review_id = client.post(f"/v1/patients/{patient_id}/reviews").json()["review_id"]
            payload = request(2).model_dump(mode="json")
            evaluated = client.post(f"/v1/reviews/{review_id}/evaluate", json=payload)
            assert evaluated.status_code == 200
            assert evaluated.json() == client.post(f"/v1/reviews/{review_id}/evaluate", json=payload).json()
            findings = client.get(f"/v1/reviews/{review_id}/findings").json()
            finding_id = next(f["finding_id"] for f in findings if f["domain"] == "hypertension")
            assert client.get(f"/v1/findings/{finding_id}").json()["finding_id"] == finding_id
            assert client.post(f"/v1/findings/{finding_id}/actions", json={"action_type": "accept"}).status_code == 201
            assert len(client.get(f"/v1/findings/{finding_id}/actions").json()) == 1
            invalid_action = client.post(f"/v1/findings/{finding_id}/actions", json={"action_type": "invalid"})
            assert invalid_action.status_code == 422 and invalid_action.json()["detail"]["code"] == "invalid_action"
            assert client.get("/v1/findings/missing").status_code == 404
            assert client.get("/v1/reviews/missing/findings").status_code == 404
            invalid_context = client.post(
                f"/v1/reviews/{review_id}/evaluate",
                json={
                    "as_of": "2026-09-24",
                    "record_coverage": {
                        "start_date": "2026-09-25",
                        "end_date": "2026-09-24",
                        "complete_resource_types": ["Encounter"],
                    },
                },
            )
            assert (
                invalid_context.status_code == 422
                and invalid_context.json()["detail"]["code"] == "invalid_evaluation_context"
            )
            client.post(f"/v1/reviews/{review_id}/complete")
            assert client.post(f"/v1/reviews/{review_id}/evaluate", json=payload).status_code == 409
            with sqlite3.connect(repo.path) as db:
                db.execute("UPDATE clinical_findings SET context_hash='wrong' WHERE finding_id=?", (finding_id,))
            storage_error = client.get(f"/v1/findings/{finding_id}")
            assert storage_error.status_code == 503 and storage_error.json()["detail"]["code"] == "storage_error"
    finally:
        app.dependency_overrides.clear()


def test_finding_integrity_read(tmp_path):
    repo, review_id = setup_review(tmp_path, 2)
    finding = ClinicalReviewEngine(repo).evaluate(review_id, request(2))[0]
    with sqlite3.connect(repo.path) as db:
        db.execute("UPDATE clinical_findings SET context_hash='wrong' WHERE finding_id=?", (finding.finding_id,))
    with pytest.raises(ValueError, match="mismatch"):
        repo.list_findings(review_id)


@pytest.mark.parametrize(
    "as_of,event_day,expected",
    [
        ("2026-09-24", "2025-09-24", FindingStatus.SATISFIED),
        ("2026-09-24", "2025-09-23", FindingStatus.POTENTIAL_CARE_GAP),
        ("2028-02-29", "2027-02-28", FindingStatus.SATISFIED),
        ("2028-02-29", "2027-02-27", FindingStatus.POTENTIAL_CARE_GAP),
        ("2026-09-24", "2026-09-25", FindingStatus.INSUFFICIENT_DATA),
        ("2026-09-24", None, FindingStatus.INSUFFICIENT_DATA),
    ],
)
def test_hypertension_annual_boundaries(tmp_path, as_of, event_day, expected):
    bundle = raw(2)
    event = next(e["resource"] for e in bundle["entry"] if e["resource"].get("id") == "enc-b1")
    if event_day:
        event["period"] = {"start": event_day, "end": event_day}
    else:
        event.pop("period")
    repo = SQLitePatientRepository(tmp_path / "bounds.db")
    patient_id = repo.import_context(parse_bundle(bundle)[0])[0]
    review = repo.create_review(patient_id)
    end = date.fromisoformat(as_of)
    start = date(end.year - 1, end.month, min(end.day, 28) if end.month == 2 else end.day)
    req = EvaluationRequest(
        as_of=end,
        record_coverage=RecordCoverage(start_date=start, end_date=end, complete_resource_types=("Encounter",)),
    )
    found = next(f for f in ClinicalReviewEngine(repo).evaluate(review.review_id, req) if f.domain == "hypertension")
    assert found.status is expected


@pytest.mark.parametrize("kind", ["undated", "unsupported", "boundary"])
def test_diabetes_foot_edge_cases(tmp_path, kind):
    bundle = raw(1)
    procedure = next(e["resource"] for e in bundle["entry"] if e["resource"].get("id") == "proc-a1")
    if kind == "undated":
        procedure.pop("performedDateTime")
    elif kind == "unsupported":
        procedure["code"]["coding"][0]["code"] = "UNSUPPORTED"
    else:
        procedure["performedDateTime"] = "2025-09-24"
    found = candidate_foot_result(bundle, request(1))
    assert found.status is (FindingStatus.SATISFIED if kind == "boundary" else FindingStatus.INSUFFICIENT_DATA)


def test_unsupported_condition_code_does_not_guess(tmp_path):
    bundle = raw(1)
    condition = next(e["resource"] for e in bundle["entry"] if e["resource"]["resourceType"] == "Condition")
    condition["code"]["coding"][0]["code"] = "E11.8"
    finding = candidate_foot_result(bundle, request(1))
    assert finding.status is FindingStatus.INSUFFICIENT_DATA
    assert "condition_coding" in finding.missing_data


def test_evaluation_runs_with_network_disabled(tmp_path, monkeypatch):
    def reject_network(*_args, **_kwargs):
        raise AssertionError("Clinical rule evaluation attempted network access")

    monkeypatch.setattr(socket.socket, "connect", reject_network)
    repo, review_id = setup_review(tmp_path, 2)
    assert ClinicalReviewEngine(repo).evaluate(review_id, request(2))


def test_recent_or_unknown_diagnosis_does_not_create_annual_gap(tmp_path):
    for label, onset in (("recent", "2026-04-01"), ("unknown", None)):
        bundle = raw(2)
        condition = next(e["resource"] for e in bundle["entry"] if e["resource"]["resourceType"] == "Condition")
        if onset:
            condition["onsetDateTime"] = onset
            condition["recordedDate"] = onset
        else:
            condition.pop("onsetDateTime")
            condition.pop("recordedDate")
        repo = SQLitePatientRepository(tmp_path / label / "diagnosis.db")
        patient_id = repo.import_context(parse_bundle(bundle)[0])[0]
        review = repo.create_review(patient_id)
        finding = next(
            f for f in ClinicalReviewEngine(repo).evaluate(review.review_id, request(2)) if f.domain == "hypertension"
        )
        assert finding.status is FindingStatus.INSUFFICIENT_DATA
        assert "annual_reassessment_eligibility" in finding.missing_data
