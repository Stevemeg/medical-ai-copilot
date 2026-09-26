"""Real PostgreSQL tests; set PHASE5_TEST_DATABASE_URL to run."""

import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import func, select, text

from backend.clinical_models import ActionType, EvaluationRequest
from backend.clinical_review import ClinicalReviewEngine
from backend.clinical_review import EvaluationConflict
from backend.db import AuditEvent, AuditHead, ClinicalFindingRow, FindingActionRow, Patient, engine, session_factory
from backend.durable_audit import append_event, verify_chain
from backend.fhir_adapter import parse_bundle
from backend.operations import audit_operation
from backend.operations import rate_limit
from backend.postgres_store import IdempotencyConflict, PostgresPatientRepository
from backend.security import ActorContext
from backend.settings import get_settings

pytestmark = pytest.mark.postgres


@pytest.fixture(autouse=True)
def postgres_url(monkeypatch):
    url = os.environ.get("PHASE5_TEST_DATABASE_URL")
    if not url:
        pytest.skip("PHASE5_TEST_DATABASE_URL not set")
    monkeypatch.setenv("DATABASE_URL", url)
    get_settings.cache_clear()
    engine.cache_clear()
    session_factory.cache_clear()
    yield
    engine().dispose()
    session_factory.cache_clear()
    engine.cache_clear()
    get_settings.cache_clear()


def context():
    raw = json.loads(Path("data/synthetic_fhir/syn_pat_001.json").read_text(encoding="utf-8"))
    raw = json.loads(json.dumps(raw).replace("SYN-PAT-001", f"SYN-PAT-{uuid4().hex[:12]}"))
    return parse_bundle(raw)[0]


def test_concurrent_import_review_evaluation_action_and_audit():
    ctx = context()
    repo = PostgresPatientRepository(actor_subject="doctor-1", actor_roles=["clinician"], request_id="concurrency-test")
    with ThreadPoolExecutor(max_workers=6) as pool:
        imported = list(pool.map(repo.import_context, [ctx] * 6))
    assert len({row[0] for row in imported}) == 1
    assert sum(row[2] for row in imported) == 1
    patient_id = imported[0][0]
    review_key = f"review-{uuid4()}"
    with ThreadPoolExecutor(max_workers=4) as pool:
        reviews = list(pool.map(lambda _: repo.create_review(patient_id, idempotency_key=review_key), range(4)))
    assert len({review.review_id for review in reviews}) == 1
    review_id = reviews[0].review_id
    request = EvaluationRequest(as_of=date(2026, 9, 25))
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: ClinicalReviewEngine(repo).evaluate(review_id, request), range(4)))
    assert all([f.finding_id for f in row] == [f.finding_id for f in results[0]] for row in results)
    finding_id = results[0][0].finding_id
    action_key = f"action-{uuid4()}"
    with ThreadPoolExecutor(max_workers=4) as pool:
        actions = list(
            pool.map(lambda _: repo.add_action(finding_id, ActionType.ACCEPT, idempotency_key=action_key), range(4))
        )
    assert len({a.action_id for a in actions}) == 1
    with session_factory()() as session:
        assert (
            session.scalar(
                select(func.count())
                .select_from(Patient)
                .where(Patient.source_patient_id == ctx.patient.source_patient_id)
            )
            == 1
        )
        assert session.scalar(
            select(func.count())
            .select_from(ClinicalFindingRow)
            .where(ClinicalFindingRow.review_id == reviews[0].review_id)
        ) == len(results[0])
        assert (
            session.scalar(
                select(func.count()).select_from(FindingActionRow).where(FindingActionRow.finding_id == finding_id)
            )
            == 1
        )
        assert verify_chain(session)["valid"]
    with pytest.raises(IdempotencyConflict):
        repo.add_action(finding_id, ActionType.DISMISS, idempotency_key=action_key)


def test_audit_concurrent_chain_and_corruption_detection():
    actor = ActorContext("audit-test", frozenset({"clinician"}), "test", True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda i: audit_operation("test_event", actor, f"parallel-{i}"), range(20)))
    with session_factory()() as session:
        assert verify_chain(session)["valid"]
        event = session.scalar(select(AuditEvent).order_by(AuditEvent.sequence_number.desc()))
        assert event is not None
        sequence = event.sequence_number
    # Disable the immutable trigger only inside a rollback-only test transaction.
    with engine().connect() as connection:
        transaction = connection.begin()
        connection.execute(text("ALTER TABLE audit_events DISABLE TRIGGER audit_events_append_only"))
        connection.execute(
            text("UPDATE audit_events SET event_payload = CAST(:payload AS jsonb) WHERE sequence_number = :seq"),
            {"payload": '{"tampered":true}', "seq": sequence},
        )
        from sqlalchemy.orm import Session

        with Session(connection) as session:
            assert not verify_chain(session)["valid"]
        transaction.rollback()
    with session_factory()() as session:
        assert verify_chain(session)["valid"]


def test_audit_failure_rolls_back_patient_import(monkeypatch):
    ctx = context()
    repo = PostgresPatientRepository()

    def fail(*args, **kwargs):
        raise RuntimeError("simulated audit failure")

    monkeypatch.setattr("backend.postgres_store.append_event", fail)
    with pytest.raises(RuntimeError):
        repo.import_context(ctx)
    with session_factory()() as session:
        assert (
            session.scalar(
                select(func.count())
                .select_from(Patient)
                .where(Patient.source_patient_id == ctx.patient.source_patient_id)
            )
            == 0
        )


@pytest.mark.parametrize("mutation", ["previous_hash", "event_hmac", "sequence_gap"])
def test_audit_verifier_rejects_other_corruption(mutation):
    with session_factory()() as session:
        sequence = session.scalar(select(func.max(AuditEvent.sequence_number)))
    assert sequence is not None
    with engine().connect() as connection:
        transaction = connection.begin()
        connection.execute(text("ALTER TABLE audit_events DISABLE TRIGGER audit_events_append_only"))
        if mutation == "sequence_gap":
            connection.execute(text("DELETE FROM audit_events WHERE sequence_number=:seq"), {"seq": sequence})
        else:
            connection.execute(
                text(f"UPDATE audit_events SET {mutation}=:bad WHERE sequence_number=:seq"),
                {"bad": "f" * 64, "seq": sequence},
            )
        from sqlalchemy.orm import Session

        with Session(connection) as session:
            assert not verify_chain(session)["valid"]
        transaction.rollback()


def test_checkpoint_mismatch_detected():
    with session_factory().begin() as session:
        head = session.get(AuditHead, 1)
        assert head is not None
        to_add = 100 - (head.last_sequence % 100)
        checkpoint_sequence = head.last_sequence + to_add
        for i in range(to_add):
            append_event(session, event_type="checkpoint_test", resource_type="test", resource_id=str(i))
    with session_factory()() as session:
        assert verify_chain(session)["valid"]
    with engine().connect() as connection:
        transaction = connection.begin()
        connection.execute(text("ALTER TABLE audit_checkpoints DISABLE TRIGGER audit_checkpoints_append_only"))
        connection.execute(
            text("UPDATE audit_checkpoints SET checkpoint_hmac=:bad WHERE last_sequence=:seq"),
            {"bad": "f" * 64, "seq": checkpoint_sequence},
        )
        from sqlalchemy.orm import Session

        with Session(connection) as session:
            assert not verify_chain(session)["valid"]
        transaction.rollback()


def test_idempotent_import_and_postgres_rate_limit():
    ctx = context()
    repo = PostgresPatientRepository(actor_subject=f"doctor-{uuid4()}")
    key = f"import-{uuid4()}"
    first = repo.import_context(ctx, idempotency_key=key)
    assert repo.import_context(ctx, idempotency_key=key) == first
    other = context()
    with pytest.raises(IdempotencyConflict):
        repo.import_context(other, idempotency_key=key)
    actor = ActorContext(repo.actor_subject, frozenset({"clinician"}), "test", True)
    from backend.settings import get_settings

    original_limit = get_settings().rate_limit_per_minute
    get_settings().rate_limit_per_minute = 2
    try:
        assert rate_limit(actor, "test_rate")
        assert rate_limit(actor, "test_rate")
        assert not rate_limit(actor, "test_rate")
    finally:
        get_settings().rate_limit_per_minute = original_limit


def test_health_and_metrics(monkeypatch):
    import api_server
    from fastapi.testclient import TestClient

    with TestClient(api_server.app) as client:
        assert client.get("/health/live").status_code == 200
        assert client.get("/health/ready").status_code == 200
        assert client.get("/v1/patients/not-a-uuid").status_code == 404
        metrics = client.get("/metrics")
        assert metrics.status_code == 200 and "medical_http_requests_total" in metrics.text
        assert "SYN-PAT" not in metrics.text
        monkeypatch.setattr(api_server, "validate_manifest", lambda *args: (_ for _ in ()).throw(ValueError()))
        assert client.get("/health/ready").status_code == 503
        monkeypatch.setattr(api_server, "engine", lambda: (_ for _ in ()).throw(ConnectionError()))
        assert client.get("/health/ready").status_code == 503


def test_relational_restrictions_and_immutable_snapshot():
    ctx = context()
    repo = PostgresPatientRepository()
    patient_id, _, _ = repo.import_context(ctx)
    repo.create_review(patient_id)
    with engine().connect() as connection:
        txn = connection.begin()
        from sqlalchemy.exc import DBAPIError

        with pytest.raises(DBAPIError):
            connection.execute(text("DELETE FROM patients WHERE patient_id=:id"), {"id": patient_id})
        txn.rollback()
    with engine().connect() as connection:
        txn = connection.begin()
        with pytest.raises(DBAPIError):
            connection.execute(
                text("UPDATE patient_snapshots SET context_hash=:bad WHERE patient_id=:id"),
                {"bad": "f" * 64, "id": patient_id},
            )
        txn.rollback()


def test_concurrent_conflicting_evaluation_reports_conflict():
    ctx = context()
    repo = PostgresPatientRepository()
    patient_id, _, _ = repo.import_context(ctx)
    review = repo.create_review(patient_id)
    requests = [EvaluationRequest(as_of=date(2026, 9, 24)), EvaluationRequest(as_of=date(2026, 9, 25))]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(ClinicalReviewEngine(repo).evaluate, review.review_id, req) for req in requests]
        outcomes = []
        for future in futures:
            try:
                outcomes.append(future.result())
            except EvaluationConflict:
                outcomes.append("conflict")
    assert outcomes.count("conflict") == 1
    assert len(repo.list_findings(review.review_id)) == len(next(x for x in outcomes if x != "conflict"))
