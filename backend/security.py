"""JWT actor verification and server-side role policy."""

from dataclasses import dataclass
from functools import lru_cache

import jwt
from fastapi import HTTPException, Request
from jwt import PyJWKClient

from backend.settings import get_settings

ROLES = frozenset({"clinician", "clinical_admin", "guideline_editor", "auditor"})


@dataclass(frozen=True)
class ActorContext:
    subject: str
    roles: frozenset[str]
    issuer: str
    authenticated: bool


@lru_cache
def _jwks(url: str) -> PyJWKClient:
    return PyJWKClient(url, cache_keys=True, lifespan=300, timeout=3)


def verify_token(token: str) -> ActorContext:
    settings = get_settings()
    try:
        header = jwt.get_unverified_header(token)
        algorithm = header.get("alg")
        if settings.auth_mode == "dev":
            dev_key = settings.dev_jwt_key.get_secret_value() if settings.dev_jwt_key else None
            if algorithm != "HS256" or not dev_key or len(dev_key) < 32:
                raise ValueError("Invalid development token configuration")
            claims = jwt.decode(
                token,
                dev_key,
                algorithms=["HS256"],
                issuer="medical-ai-copilot-dev",
                audience="medical-ai-copilot",
                options={"require": ["exp", "iat", "sub"]},
            )
        else:
            if algorithm not in {"RS256", "ES256"}:
                raise ValueError("Unsupported JWT algorithm")
            key = _jwks(settings.oidc_jwks_url or "").get_signing_key_from_jwt(token).key
            claims = jwt.decode(
                token,
                key,
                algorithms=[algorithm],
                issuer=settings.oidc_issuer,
                audience=settings.oidc_audience,
                options={"require": ["exp", "iat", "sub"]},
            )
        subject = claims.get("sub")
        if not isinstance(subject, str) or not subject:
            raise ValueError("Missing subject")
        roles = claims.get("roles", [])
        if not isinstance(roles, list) or not set(roles) <= ROLES:
            raise ValueError("Invalid roles")
        return ActorContext(subject, frozenset(roles), str(claims["iss"]), True)
    except (jwt.PyJWTError, ValueError, TypeError, KeyError) as exc:
        raise HTTPException(401, detail={"code": "invalid_token", "message": "Authentication failed"}) from exc


def actor_from_request(request: Request) -> ActorContext:
    header = request.headers.get("authorization", "")
    if header.startswith("Bearer "):
        return verify_token(header[7:])
    if get_settings().auth_mode == "dev":
        return ActorContext("development-user", frozenset({"clinician"}), "local-development", False)
    raise HTTPException(401, detail={"code": "authentication_required", "message": "Authentication required"})


def required_roles(path: str, method: str) -> frozenset[str] | None:
    if path == "/metrics" and get_settings().app_env == "production":
        return frozenset({"auditor", "clinical_admin"})
    if path.startswith("/health/") or path in {"/api/health", "/metrics", "/api/examples", "/api/sources"}:
        return None
    if path.startswith("/v1/audit"):
        return frozenset({"auditor", "clinical_admin"})
    if path in {"/v1/knowledge-sources", "/v1/evidence/query", "/api/ask"}:
        return frozenset({"clinician", "clinical_admin", "guideline_editor"})
    if path == "/v1/demo-patients":
        return frozenset({"clinician", "clinical_admin"})
    if path.startswith("/v1/"):
        return frozenset({"clinician", "clinical_admin"})
    return None
