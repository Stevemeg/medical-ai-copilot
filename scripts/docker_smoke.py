"""Deterministic in-container API smoke without a live model provider."""

import json
from datetime import datetime, timedelta, timezone

import jwt
import numpy as np
from fastapi.testclient import TestClient

from api_server import app
from backend.db import session_factory
from backend.durable_audit import verify_chain
from backend.evidence_models import SupportStatus
from backend.evidence_pipeline import EvidenceAnswerService
from backend.settings import get_settings
from embeddings.retrieve import EvidenceRetriever
from embeddings.reranker import RRFFallbackReranker
import backend.rag_pipeline as rag_pipeline


def token(role: str) -> str:
    now = datetime.now(timezone.utc)
    return jwt.encode(
        {
            "iss": "medical-ai-copilot-dev",
            "aud": "medical-ai-copilot",
            "sub": f"smoke-{role}",
            "iat": now,
            "nbf": now,
            "exp": now + timedelta(minutes=10),
            "roles": [role],
        },
        get_settings().dev_jwt_key.get_secret_value(),
        algorithm="HS256",
    )


class FixedEmbedder:
    def __init__(self, vector):
        self.vector = vector

    def encode(self, sentences, *, normalize_embeddings):
        return np.asarray([self.vector], dtype="float32")


class QuoteGenerator:
    def generate(self, system_prompt, user_prompt):
        rows = json.loads(
            user_prompt.split("<UNTRUSTED_EVIDENCE_JSON>\n", 1)[1].split("\n</UNTRUSTED_EVIDENCE_JSON>", 1)[0]
        )
        unit = rows[0]
        return json.dumps(
            {
                "status": "grounded",
                "claims": [
                    {"claim_id": "smoke-claim", "text": unit["text"][:80], "evidence_ids": [unit["evidence_unit_id"]]}
                ],
            }
        )


class QuoteVerifier:
    def verify(self, claim, evidence):
        return (
            SupportStatus.SUPPORTED if all(claim.text in unit.text for unit in evidence) else SupportStatus.UNSUPPORTED
        )


class UnavailableVerifier:
    def verify(self, claim, evidence):
        return SupportStatus.UNCERTAIN


def main():
    if get_settings().auth_mode != "dev":
        raise RuntimeError("Smoke fixture requires explicit development authentication")
    headers = {"Authorization": f"Bearer {token('clinician')}"}
    with TestClient(app) as client:
        assert client.get("/health/live").status_code == 200
        assert client.get("/health/ready").status_code == 200
        imported = client.post("/v1/demo-patients/syn_pat_001/load", headers=headers)
        assert imported.status_code == 200, imported.text
        patient_id = imported.json()["patient_id"]
        assert client.get(f"/v1/patients/{patient_id}", headers=headers).status_code == 200
        review = client.post(
            f"/v1/patients/{patient_id}/reviews", headers={**headers, "Idempotency-Key": "smoke-review"}
        )
        assert review.status_code == 201, review.text
        review_id = review.json()["review_id"]
        evaluated = client.post(f"/v1/reviews/{review_id}/evaluate", json={"as_of": "2026-09-25"}, headers=headers)
        assert evaluated.status_code == 200 and evaluated.json(), evaluated.text
        findings = client.get(f"/v1/reviews/{review_id}/findings", headers=headers).json()
        finding_id = findings[0]["finding_id"]
        action = client.post(
            f"/v1/findings/{finding_id}/actions",
            json={"action_type": "accept"},
            headers={**headers, "Idempotency-Key": "smoke-action"},
        )
        assert action.status_code == 201, action.text
        repeated = client.post(
            f"/v1/findings/{finding_id}/actions",
            json={"action_type": "accept"},
            headers={**headers, "Idempotency-Key": "smoke-action"},
        )
        assert repeated.json()["action_id"] == action.json()["action_id"]
        assert (
            client.post(
                "/v1/patients/import", json={}, headers={"Authorization": f"Bearer {token('auditor')}"}
            ).status_code
            == 403
        )
        assert client.get("/v1/patients", headers={"Authorization": "Bearer invalid"}).status_code == 401
        retriever = EvidenceRetriever(use_reranker=False)
        index = next(
            i
            for i, unit in enumerate(retriever.units)
            if unit.document_id == "nice-ng136" and unit.page_start is not None
        )
        retriever.embedder = FixedEmbedder(retriever.vectors[index])
        retriever.reranker = RRFFallbackReranker()
        rag_pipeline._service = EvidenceAnswerService(retriever, QuoteGenerator(), QuoteVerifier())
        query = {"query": "How should hypertension be diagnosed?", "document_ids": ["nice-ng136"], "top_k": 5}
        grounded = client.post("/v1/evidence/query", json=query, headers=headers)
        assert grounded.status_code == 200 and grounded.json()["status"] == "grounded", grounded.text
        rag_pipeline._service = EvidenceAnswerService(retriever, QuoteGenerator(), UnavailableVerifier())
        uncertain = client.post("/v1/evidence/query", json=query, headers=headers)
        assert uncertain.status_code == 200 and uncertain.json()["status"] == "abstained"
        assert uncertain.json()["claims"] == []
    with session_factory()() as session:
        assert verify_chain(session)["valid"]
    print("Docker smoke PASS: API, Postgres, review, action, evidence, fail-closed grounding, audit")


if __name__ == "__main__":
    main()
