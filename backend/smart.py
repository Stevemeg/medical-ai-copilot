"""SMART 2.2 public client: browser-bound single-use state, S256, no stored tokens."""

import base64
import hashlib
import hmac
import secrets
from datetime import timedelta
from urllib.parse import urlencode, urlsplit

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import RedirectResponse
from sqlalchemy import select

from backend.db import SmartLaunch, transaction, utcnow
from backend.fhir_adapter import parse_bundle
from backend.fhir_client import FHIRClient, RESOURCE_TYPES
from backend.patient_api import get_repository
from backend.patient_models import PatientContext
from backend.postgres_store import PostgresPatientRepository
from backend.safe_http import AcquisitionError, SafeHTTP, origin
from backend.settings import get_settings

router = APIRouter()


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def challenge(verifier: str) -> str:
    return base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip("=")


def smart_http() -> SafeHTTP:
    settings = get_settings()
    return SafeHTTP({origin(url) for url in settings.smart_issuers + settings.smart_endpoint_origins})


def validate_redirect() -> str:
    settings = get_settings()
    url = urlsplit(settings.smart_redirect_uri)
    if url.query or url.fragment or url.username or url.password or url.path != "/smart/callback":
        raise AcquisitionError("Invalid registered redirect URI")
    if url.scheme != "https" and not (
        settings.app_env != "production" and url.scheme == "http" and url.hostname in {"localhost", "127.0.0.1"}
    ):
        raise AcquisitionError("Redirect URI requires HTTPS")
    return settings.smart_redirect_uri


def discover(issuer: str, http: SafeHTTP) -> dict:
    settings = get_settings()
    if issuer not in settings.smart_issuers:
        raise AcquisitionError("Unregistered FHIR issuer")
    metadata = http.json(issuer.rstrip("/") + "/.well-known/smart-configuration")
    for key in ("authorization_endpoint", "token_endpoint"):
        endpoint = metadata.get(key, "")
        if origin(endpoint) not in http.allowed_origins or urlsplit(endpoint).query:
            raise AcquisitionError("Unregistered OAuth endpoint")
    if "S256" not in metadata.get("code_challenge_methods_supported", []):
        raise AcquisitionError("SMART server does not advertise S256")
    return metadata


@router.get("/smart/launch")
def launch(iss: str, launch: str, http: SafeHTTP = Depends(smart_http)):
    try:
        if not 1 <= len(launch) <= 1024:
            raise AcquisitionError("Invalid launch context")
        redirect = validate_redirect()
        metadata = discover(iss, http)
    except AcquisitionError:
        raise HTTPException(400, "SMART launch configuration rejected") from None
    state, browser, verifier = secrets.token_urlsafe(32), secrets.token_urlsafe(32), secrets.token_urlsafe(64)
    settings = get_settings()
    with transaction() as session:
        session.add(
            SmartLaunch(
                state_hash=digest(state),
                browser_hash=digest(browser),
                issuer=iss,
                token_endpoint=metadata["token_endpoint"],
                verifier=verifier,
                expires_at=utcnow() + timedelta(seconds=settings.smart_state_seconds),
                consumed=False,
            )
        )
    scope = "launch patient/Patient.r " + " ".join(f"patient/{kind}.rs" for kind in RESOURCE_TYPES)
    params = dict(
        response_type="code",
        client_id=settings.smart_client_id,
        redirect_uri=redirect,
        launch=launch,
        scope=scope,
        state=state,
        aud=iss,
        code_challenge=challenge(verifier),
        code_challenge_method="S256",
    )
    response = RedirectResponse(metadata["authorization_endpoint"] + "?" + urlencode(params), status_code=302)
    response.set_cookie(
        "smart_browser",
        browser,
        httponly=True,
        secure=redirect.startswith("https:"),
        samesite="lax",
        max_age=settings.smart_state_seconds,
        path="/",
    )
    response.headers["Cache-Control"] = "no-store"
    return response


def consume(state: str, browser: str) -> tuple[str, str, str]:
    with transaction() as session:
        row = session.scalar(select(SmartLaunch).where(SmartLaunch.state_hash == digest(state)).with_for_update())
        if (
            row is None
            or row.consumed
            or row.expires_at <= utcnow()
            or not browser
            or not hmac.compare_digest(row.browser_hash, digest(browser))
            or not row.verifier
        ):
            raise HTTPException(400, "Invalid or expired SMART state")
        result = row.issuer, row.token_endpoint, row.verifier
        row.consumed, row.verifier = True, None
        return result


@router.get("/smart/callback")
def callback(request: Request, state: str, code: str, http: SafeHTTP = Depends(smart_http)):
    if len(state) > 128 or not 1 <= len(code) <= 2048:
        raise HTTPException(400, "Invalid OAuth response")
    issuer, endpoint, verifier = consume(state, request.cookies.get("smart_browser", ""))
    settings = get_settings()
    try:
        if issuer not in settings.smart_issuers:
            raise AcquisitionError("Issuer registration changed")
        token = http.json(
            endpoint,
            method="POST",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            body=urlencode(
                dict(
                    grant_type="authorization_code",
                    code=code,
                    code_verifier=verifier,
                    client_id=settings.smart_client_id,
                    redirect_uri=validate_redirect(),
                )
            ).encode(),
        )
        if token.get("token_type", "").lower() != "bearer" or not token.get("patient") or not token.get("access_token"):
            raise AcquisitionError("Missing token or patient context")
        bundle = FHIRClient(issuer, token["access_token"], http).bundle(token["patient"])
        context, _ = parse_bundle(bundle)
        with transaction() as session:
            row = session.get(SmartLaunch, digest(state))
            if row is None or row.expires_at <= utcnow():
                raise AcquisitionError("Launch expired during import")
            row.context_json = context.model_dump(mode="json")
    except (ValueError, KeyError, TypeError):
        raise HTTPException(400, "SMART authorization or FHIR context failed") from None
    # Only a one-time context handle reaches the UI. No code or token is returned.
    response = RedirectResponse("/?smart=" + digest(state), status_code=303)
    response.headers["Cache-Control"] = "no-store"
    return response


@router.post("/v1/smart/{handle}/import")
def import_context(handle: str, request: Request, repository: PostgresPatientRepository = Depends(get_repository)):
    with transaction() as session:
        row = session.scalar(select(SmartLaunch).where(SmartLaunch.state_hash == handle).with_for_update())
        browser = request.cookies.get("smart_browser", "")
        if (
            row is None
            or row.expires_at <= utcnow()
            or not row.context_json
            or not browser
            or not hmac.compare_digest(row.browser_hash, digest(browser))
        ):
            raise HTTPException(400, "SMART context unavailable")
        context = PatientContext.model_validate(row.context_json)
        # Deterministic idempotency means a failed response can safely be retried.
        patient_id, _, _ = repository.import_context(context, idempotency_key="smart-" + handle)
        row.context_json = None
    return {"patient_id": patient_id, "context_source": "SMART launch / FHIR server"}
