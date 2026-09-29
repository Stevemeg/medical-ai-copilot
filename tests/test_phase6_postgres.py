"""Real PostgreSQL governance, SMART replay and retention regression tests."""

from datetime import timedelta
from urllib.parse import parse_qs, urlsplit
from uuid import uuid4

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select, func

from backend.db import (
    AuditEvent,
    GuidelineCandidate,
    IdempotencyKey,
    RateBucket,
    SmartLaunch,
    session_factory,
    transaction,
    utcnow,
)
from backend.durable_audit import verify_chain
from backend.guideline_updates import CandidateInput, active_overlay, decide, prepare, register_candidate
from backend.knowledge_models import RetrievalPolicy
from backend.retention import cleanup
from backend.security import ActorContext
from backend.settings import get_settings
from backend.smart import digest, smart_http
from backend.source_registry import SourceRegistry
from tests.test_phase5_postgres import postgres_url  # noqa: F401
from tests.test_phase6_interop import MockFHIR

pytestmark = pytest.mark.postgres
EDITOR = ActorContext("synthetic-editor", frozenset({"guideline_editor"}), "test", True)


class Embedder:
    def encode(self, texts, **kwargs):
        matrix = np.zeros((len(texts), 384), dtype="float32")
        matrix[:, 0] = 1
        return matrix


def test_guideline_quarantine_diff_activation_history_and_audit(monkeypatch):
    source = CandidateInput(
        document_id="nice-ng136",
        version_id="test-release-" + uuid4().hex,
        source_url="https://www.nice.org.uk/guidance/ng136",
        updated_at="2026-09-29",
        license_note="Entirely synthetic isolated database test; not publisher content or clinical guidance.",
        recommendations=[
            {
                "recommendation_id": "1.4.24",
                "section": "1.4",
                "page": 1,
                "text": "Synthetic fixture: display a blue review marker for engineering test " + uuid4().hex,
            }
        ],
    )
    candidate_id = register_candidate(source, EDITOR, "test")
    registry = SourceRegistry()
    overlay, units, _ = active_overlay(registry)
    assert source.version_id not in overlay.versions
    with pytest.raises(ValueError):
        decide(candidate_id, "activate", "Not yet approved", EDITOR, "test")
    prepare(candidate_id, EDITOR, "test", Embedder())
    with session_factory()() as session:
        candidate = session.get(GuidelineCandidate, __import__("uuid").UUID(candidate_id))
        assert candidate.status == "review_required"
        assert candidate.diff["modified"]
        assert any(r["rule_id"] == "HTN_ANNUAL_CARE_REVIEW" for r in candidate.diff["affected_rules"])
    with pytest.raises(PermissionError):
        decide(
            candidate_id,
            "approve",
            "Unauthorized attempt",
            ActorContext("clinician", frozenset({"clinician"}), "test", True),
            "test",
        )
    decide(candidate_id, "approve", "Reviewed synthetic diff", EDITOR, "test")
    overlay, _, _ = active_overlay(registry)
    assert source.version_id not in overlay.versions
    decide(candidate_id, "activate", "Activate isolated test release", EDITOR, "test")
    overlay, units, vectors = active_overlay(registry)
    assert overlay.eligible({"version_id": source.version_id}, RetrievalPolicy.CURRENT_CLINICAL)
    assert overlay.eligible({"version_id": "nice-ng136-2026-02-26"}, RetrievalPolicy.HISTORICAL_ONLY)
    assert not overlay.eligible({"version_id": "nice-ng136-2026-02-26"}, RetrievalPolicy.CURRENT_CLINICAL)
    assert len(units) >= 1 and vectors.shape[1] == 384
    monkeypatch.setattr(get_settings(), "guideline_updates_enabled", True)
    from backend.evidence_models import RetrievalRequest, QueryIntent
    from embeddings.retrieve import EvidenceRetriever

    retriever = EvidenceRetriever(embedder=Embedder(), use_reranker=False)
    current = retriever.retrieve(RetrievalRequest(query="Synthetic blue review marker", document_ids=["nice-ng136"]))
    assert current.evidence and all(r.unit.version_id == source.version_id for r in current.evidence)
    old_index = next(
        i
        for i, u in enumerate(retriever.units)
        if u.version_id == "nice-ng136-2026-02-26" and u.recommendation_id == "1.4.24"
    )
    monkeypatch.setattr(retriever.embedder, "encode", lambda *a, **k: retriever.vectors[old_index : old_index + 1])
    historical = retriever.retrieve(
        RetrievalRequest(
            query=retriever.units[old_index].text[:500],
            intent=QueryIntent.HISTORICAL,
            version_ids=["nice-ng136-2026-02-26"],
        )
    )
    assert historical.evidence and all(r.unit.lifecycle_status.value == "superseded" for r in historical.evidence)
    from backend.clinical_rules import default_registry

    assert (
        next(
            r for r in default_registry().current_rules() if r.definition.domain == "hypertension"
        ).definition.status.value
        == "disabled"
    )
    with session_factory()() as session:
        assert verify_chain(session)["valid"]
        assert session.scalar(
            select(AuditEvent).where(
                AuditEvent.resource_id == candidate_id, AuditEvent.event_type == "guideline_activated"
            )
        )
    from sqlalchemy import text
    from sqlalchemy.exc import DBAPIError

    with pytest.raises(DBAPIError), transaction() as session:
        session.execute(
            text("UPDATE guideline_candidates SET status='quarantined' WHERE candidate_id=:id"), {"id": candidate_id}
        )


def test_retention_dry_run_and_execute_preserve_history():
    actor = "retention-" + uuid4().hex
    now = utcnow()
    with transaction() as session:
        for age in (timedelta(days=20), timedelta(minutes=0)):
            session.add(RateBucket(actor_subject=actor, operation="test", minute=now - age, count=1))
            session.add(
                IdempotencyKey(
                    actor_subject=actor,
                    operation="test",
                    key=str(age),
                    request_fingerprint="a" * 64,
                    response_json={},
                    created_at=now - age,
                )
            )
        before = session.scalar(select(func.count()).select_from(AuditEvent))
    with transaction() as session:
        dry = cleanup(session, get_settings())
        assert dry["rate_buckets"] >= 1 and dry["idempotency_keys"] >= 1
        assert (
            session.scalar(select(func.count()).select_from(RateBucket).where(RateBucket.actor_subject == actor)) == 2
        )
    with transaction() as session:
        result = cleanup(session, get_settings(), execute=True)
        assert result["rate_buckets"] >= 1 and result["idempotency_keys"] >= 1
    with session_factory()() as session:
        assert (
            session.scalar(select(func.count()).select_from(RateBucket).where(RateBucket.actor_subject == actor)) == 1
        )
        assert (
            session.scalar(
                select(func.count()).select_from(IdempotencyKey).where(IdempotencyKey.actor_subject == actor)
            )
            == 1
        )
        assert session.scalar(select(func.count()).select_from(AuditEvent)) == before
        assert verify_chain(session)["valid"]


class OAuth(MockFHIR):
    def __init__(self):
        super().__init__()
        self.fail = None
        self.verifier = None

    def json(self, url, **kwargs):
        if url.endswith("smart-configuration"):
            return {
                "authorization_endpoint": "https://ehr.test/auth",
                "token_endpoint": "https://ehr.test/token",
                "code_challenge_methods_supported": ["S256"],
            }
        if url.endswith("/token"):
            from backend.smart import challenge

            params = parse_qs(kwargs["body"].decode())
            if self.fail == "token" or challenge(params["code_verifier"][0]) != self.verifier:
                raise ValueError("PKCE/token rejection")
            return {
                "token_type": "Bearer",
                "access_token": "SECRET_SYNTHETIC_ACCESS",
                "patient": None if self.fail == "patient" else "SYN-PAT-002",
            }
        return super().json(url, **kwargs)


@pytest.mark.parametrize("attack", [None, "wrong_state", "expired", "wrong_browser", "pkce", "token", "patient"])
def test_smart_roundtrip_and_failures(monkeypatch, attack):
    import api_server

    settings = get_settings()
    monkeypatch.setattr(settings, "smart_issuers", ["https://ehr.test/fhir"])
    monkeypatch.setattr(settings, "smart_endpoint_origins", ["https://ehr.test"])
    http = OAuth()
    api_server.app.dependency_overrides[smart_http] = lambda: http
    try:
        with TestClient(api_server.app, base_url="http://localhost:18000") as client:
            response = client.get(
                "/smart/launch",
                params={"iss": "https://ehr.test/fhir", "launch": "test-context"},
                follow_redirects=False,
            )
            assert response.status_code == 302, response.text
            params = parse_qs(urlsplit(response.headers["location"]).query)
            state = params["state"][0]
            http.verifier = params["code_challenge"][0]
            if attack == "wrong_state":
                state = "wrong-state"
            if attack == "wrong_browser":
                client.cookies.clear()
            if attack in {"expired", "pkce"}:
                with transaction() as session:
                    row = session.get(SmartLaunch, digest(state))
                    if attack == "expired":
                        row.expires_at = utcnow() - timedelta(seconds=1)
                    else:
                        row.verifier = "wrong-verifier"
            http.fail = attack
            result = client.get(
                "/smart/callback", params={"state": state, "code": "SECRET_SYNTHETIC_CODE"}, follow_redirects=False
            )
            assert result.status_code == (303 if attack is None else 400), result.text
            assert "SECRET_SYNTHETIC" not in result.text
            if attack is None:
                imported = client.post(f"/v1/smart/{digest(state)}/import")
                assert imported.status_code == 200, imported.text
                assert imported.json()["context_source"] == "SMART launch / FHIR server"
                assert client.post(f"/v1/smart/{digest(state)}/import").status_code == 400
                assert (
                    client.get(
                        "/smart/callback", params={"state": state, "code": "replay"}, follow_redirects=False
                    ).status_code
                    == 400
                )
    finally:
        api_server.app.dependency_overrides.clear()


def candidate_source():
    return CandidateInput(
        document_id="nice-ng136",
        version_id="test-release-" + uuid4().hex,
        source_url="https://www.nice.org.uk/guidance/ng136",
        updated_at="2026-09-29",
        license_note="Synthetic engineering test only; never clinical guidance.",
        recommendations=[
            {
                "recommendation_id": "1.4.24",
                "section": "1.4",
                "page": 1,
                "text": "Synthetic fixture: display a blue review marker.",
            }
        ],
    )


def test_concurrent_activation_rejects_obsolete_base_and_audit_failure_rolls_back(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import backend.guideline_updates as governance
    from backend.db import GuidelineActivation
    from uuid import UUID

    ids = [register_candidate(candidate_source(), EDITOR, "test") for _ in range(2)]
    for ident in ids:
        prepare(ident, EDITOR, "test", Embedder())
        decide(ident, "approve", "Reviewed synthetic candidate", EDITOR, "test")

    original = governance._audit

    def failed_audit(*args):
        raise RuntimeError("Simulated audit outage")

    monkeypatch.setattr(governance, "_audit", failed_audit)
    with pytest.raises(RuntimeError):
        decide(ids[0], "activate", "Test atomic rollback", EDITOR, "test")
    with session_factory()() as session:
        assert session.get(GuidelineCandidate, UUID(ids[0])).status == "approved"
        active = session.get(GuidelineActivation, "nice-ng136")
        assert not active or str(active.candidate_id) not in ids
    monkeypatch.setattr(governance, "_audit", original)

    def activate(ident):
        try:
            decide(ident, "activate", "Concurrent synthetic activation", EDITOR, "test")
            return True
        except ValueError:
            return False

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sum(pool.map(activate, ids)) == 1
    with session_factory()() as session:
        assert verify_chain(session)["valid"]


def test_guideline_http_rbac_and_review_transitions(monkeypatch):
    import api_server
    import backend.guideline_api as routes
    from backend.security import ActorContext
    from uuid import UUID

    monkeypatch.setattr(get_settings(), "guideline_updates_enabled", True)
    monkeypatch.setattr(routes, "prepare", lambda *args: prepare(*args, embedder=Embedder()))
    actor = ActorContext("synthetic-clinician", frozenset({"clinician"}), "test", True)
    monkeypatch.setattr(api_server, "actor_from_request", lambda request: actor)
    with TestClient(api_server.app) as client:
        source = candidate_source().model_dump(mode="json")
        assert client.post("/v1/guideline-updates", json=source).status_code == 403
        actor = EDITOR
        response = client.post("/v1/guideline-updates", json=source)
        assert response.status_code == 201, response.text
        ident = response.json()["candidate_id"]
        assert client.get("/v1/guidelines").status_code == 200
        assert client.get("/v1/guidelines/missing/versions").status_code == 404
        assert client.get("/v1/guideline-updates").status_code == 200
        assert client.post(f"/v1/guideline-updates/{ident}/prepare").status_code == 200
        actor = ActorContext("synthetic-clinician", frozenset({"clinician"}), "test", True)
        assert (
            client.post(
                f"/v1/guideline-updates/{ident}/activate", json={"note": "Unauthorized synthetic test"}
            ).status_code
            == 403
        )
        with session_factory()() as session:
            assert session.get(GuidelineCandidate, UUID(ident)).status == "review_required"
        actor = EDITOR
        assert (
            client.post(f"/v1/guideline-updates/{ident}/reject", json={"note": "Reject synthetic test"}).status_code
            == 200
        )
        assert (
            client.post(f"/v1/guideline-updates/{ident}/approve", json={"note": "Invalid transition"}).status_code
            == 409
        )
