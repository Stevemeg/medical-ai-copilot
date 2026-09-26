"""Offline authentication, authorization, and configuration checks."""

from datetime import datetime, timedelta, timezone

import jwt
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from backend.security import ActorContext, verify_token, required_roles
from backend.settings import Settings, get_settings


def _token(**changes):
    now = datetime.now(timezone.utc)
    claims = {
        "iss": "medical-ai-copilot-dev",
        "aud": "medical-ai-copilot",
        "sub": "doctor-1",
        "iat": now,
        "nbf": now,
        "exp": now + timedelta(minutes=5),
        "roles": ["clinician"],
    }
    claims.update(changes)
    return jwt.encode(claims, get_settings().dev_jwt_key.get_secret_value(), algorithm="HS256")


def test_dev_jwt_validation_matrix():
    from fastapi import HTTPException

    assert verify_token(_token()).subject == "doctor-1"
    bad = [
        _token(exp=datetime.now(timezone.utc) - timedelta(minutes=1)),
        _token(nbf=datetime.now(timezone.utc) + timedelta(minutes=10)),
        _token(iss="other"),
        _token(aud="other"),
        jwt.encode({"sub": "doctor-1"}, "wrong-signature-key-1234567890123456", algorithm="HS256"),
        jwt.encode({"sub": "doctor-1"}, get_settings().dev_jwt_key.get_secret_value(), algorithm="HS512"),
        jwt.encode({"sub": "doctor-1"}, key="", algorithm="none"),
    ]
    for token in bad:
        with pytest.raises(HTTPException) as error:
            verify_token(token)
        assert error.value.status_code == 401


@pytest.mark.parametrize(
    "path,method",
    [
        ("/v1/patients", "GET"),
        ("/v1/patients/import", "POST"),
        ("/v1/patients/abc/reviews", "POST"),
        ("/v1/reviews/abc/evaluate", "POST"),
        ("/v1/findings/abc/actions", "POST"),
        ("/v1/evidence/query", "POST"),
    ],
)
def test_auditor_cannot_use_clinical_endpoints(monkeypatch, path, method):
    from api_server import app
    import api_server

    monkeypatch.setattr(
        api_server, "actor_from_request", lambda _: ActorContext("audit-1", frozenset({"auditor"}), "test", True)
    )
    with TestClient(app) as client:
        response = client.request(method, path)
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "forbidden"


def test_production_configuration_fails_closed():
    base = {
        "app_env": "production",
        "database_url": "postgresql+psycopg://u:p@localhost/db",
        "auth_mode": "oidc",
        "audit_hmac_key": "x" * 32,
        "oidc_issuer": "https://issuer.example",
        "oidc_audience": "medical-api",
        "oidc_jwks_url": "https://issuer.example/jwks",
    }
    Settings(**base, cors_origins=["https://app.example"], _env_file=None)
    for change in (
        {"database_url": ""},
        {"oidc_issuer": None},
        {"oidc_jwks_url": "http://issuer.example/jwks"},
        {"auth_mode": "dev"},
        {"cors_origins": ["*"]},
        {"audit_hmac_key": "short"},
        {"audit_hmac_key": "replace-with-a-real-production-key-please"},
    ):
        with pytest.raises(ValidationError):
            Settings(**{**base, **change}, _env_file=None)


@pytest.mark.parametrize(
    "role,patient,evidence,audit",
    [
        ("clinician", True, True, False),
        ("clinical_admin", True, True, True),
        ("guideline_editor", False, True, False),
        ("auditor", False, False, True),
    ],
)
def test_permission_matrix(role, patient, evidence, audit):
    assert (role in required_roles("/v1/patients/import", "POST")) is patient
    assert (role in required_roles("/v1/reviews/any/evaluate", "POST")) is patient
    assert (role in required_roles("/v1/findings/any/actions", "POST")) is patient
    assert (role in required_roles("/v1/evidence/query", "POST")) is evidence
    assert (role in required_roles("/v1/audit/events", "GET")) is audit


def test_unauthenticated_request_returns_401(monkeypatch):
    import api_server
    from fastapi import HTTPException

    def no_actor(_):
        raise HTTPException(401, detail={"code": "authentication_required", "message": "Authentication required"})

    monkeypatch.setattr(api_server, "actor_from_request", no_actor)
    with TestClient(api_server.app) as client:
        response = client.get("/v1/patients")
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "authentication_required"


def test_oidc_rs256_verification_with_test_key(monkeypatch):
    from cryptography.hazmat.primitives.asymmetric import rsa
    import backend.security as security

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    settings = Settings(
        app_env="production",
        database_url="postgresql+psycopg://u:p@localhost/db",
        auth_mode="oidc",
        audit_hmac_key="x" * 32,
        oidc_issuer="https://issuer.test",
        oidc_audience="medical-api",
        oidc_jwks_url="https://issuer.test/jwks",
        cors_origins=["https://app.test"],
        _env_file=None,
    )

    class SigningKey:
        def __init__(self):
            self.key = key.public_key()

    class JWKS:
        def get_signing_key_from_jwt(self, token):
            return SigningKey()

    monkeypatch.setattr(security, "get_settings", lambda: settings)
    monkeypatch.setattr(security, "_jwks", lambda _: JWKS())
    now = datetime.now(timezone.utc)
    token = jwt.encode(
        {
            "iss": "https://issuer.test",
            "aud": "medical-api",
            "sub": "doctor-2",
            "iat": now,
            "nbf": now,
            "exp": now + timedelta(minutes=5),
            "roles": ["clinician"],
        },
        key,
        algorithm="RS256",
    )
    assert verify_token(token).subject == "doctor-2"
    wrong = jwt.encode(
        {
            "iss": "wrong",
            "aud": "medical-api",
            "sub": "doctor-2",
            "iat": now,
            "exp": now + timedelta(minutes=5),
            "roles": ["clinician"],
        },
        key,
        algorithm="RS256",
    )
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as error:
        verify_token(wrong)
    assert error.value.status_code == 401


def test_http_logging_omits_payload_token_and_keys(caplog):
    import logging
    from api_server import app

    marker = "Synthetic Patient Private Marker"
    with caplog.at_level(logging.INFO, logger="api"), TestClient(app) as client:
        client.post("/v1/fhir/validate", json={"private_marker": marker})
        client.get("/v1/patients", headers={"Authorization": "Bearer sensitive-test-token"})
    entries = "\n".join(record.getMessage() for record in caplog.records if record.name == "api")
    assert marker not in entries
    assert "sensitive-test-token" not in entries
    assert get_settings().audit_hmac_key.get_secret_value() not in entries
    assert get_settings().dev_jwt_key.get_secret_value() not in entries


def test_production_log_formatter_has_structured_correlation():
    import json
    import logging
    from backend.logging_config import JSONFormatter

    record = logging.LogRecord("api", logging.INFO, __file__, 1, "http_request", (), None)
    record.request_id = "a1b2c3d4-e5f6-4789-a123-abcdefabcdef"
    record.route = "/v1/patients/{patient_id}"
    record.status_code = 200
    result = json.loads(JSONFormatter().format(record))
    assert result["event"] == "http_request"
    assert result["request_id"] == record.request_id
    assert result["route"] == record.route


def test_request_id_validation_and_response_header():
    from api_server import app

    supplied = "a1b2c3d4-e5f6-4789-a123-abcdefabcdef"
    with TestClient(app) as client:
        good = client.get("/health/live", headers={"X-Request-ID": supplied})
        bad = client.get("/health/live", headers={"X-Request-ID": "patient-name"})
    assert good.headers["X-Request-ID"] == supplied
    assert bad.headers["X-Request-ID"] != "patient-name"


def test_provider_receives_only_correlation_id(monkeypatch):
    import groq
    import backend.generator as generator
    from backend.request_context import set_request_id, reset_request_id

    captured = {}

    class Completions:
        def create(self, **kwargs):
            return type(
                "Result", (), {"choices": [type("Choice", (), {"message": type("Message", (), {"content": "{}"})()})()]}
            )()

    class FakeGroq:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.chat = type("Chat", (), {"completions": Completions()})()

    monkeypatch.setattr(groq, "Groq", FakeGroq)
    monkeypatch.setattr(generator, "get_groq_api_key", lambda: "test-provider-key")
    token = set_request_id("a1b2c3d4-e5f6-4789-a123-abcdefabcdef")
    try:
        assert generator.GroqAnswerGenerator().generate("system", "user") == "{}"
    finally:
        reset_request_id(token)
    assert captured["default_headers"] == {"X-Request-ID": "a1b2c3d4-e5f6-4789-a123-abcdefabcdef"}
