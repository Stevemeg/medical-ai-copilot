"""Offline boundary tests: transport, SMART contracts, CDS and official profiles."""

import copy
import json
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from backend.abdm import analyze, profiles, validate_recorded_example
from backend.cds_hooks import HookRequest, cards, evaluation_context, hook_bundle
from backend.clinical_review import ClinicalReviewEngine
from backend.fhir_adapter import parse_bundle
from backend.fhir_client import FHIRClient, RESOURCE_TYPES
from backend.patient_api import get_repository
from backend.patient_store import SQLitePatientRepository
from backend.safe_http import AcquisitionError, SafeHTTP, validate_url
from backend.settings import get_settings
from backend.smart import challenge, discover


def bundle(number=2):
    return json.loads(Path(f"data/synthetic_fhir/syn_pat_{number:03}.json").read_text(encoding="utf-8"))


def prefetch(raw):
    resources = [e["resource"] for e in raw["entry"]]
    return {
        "patient": next(r for r in resources if r["resourceType"] == "Patient"),
        **{
            kind: {
                "resourceType": "Bundle",
                "type": "searchset",
                "entry": [{"resource": r} for r in resources if r["resourceType"] == kind],
            }
            for kind in RESOURCE_TYPES
        },
    }


def hook(raw=None):
    payload = prefetch(raw or bundle())
    return {
        "hook": "patient-view",
        "hookInstance": str(uuid4()),
        "context": {"userId": "Practitioner/synthetic", "patientId": payload["patient"]["id"]},
        "prefetch": payload,
    }


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1",
        "http://localhost",
        "http://169.254.169.254",
        "http://10.0.0.1",
        "http://172.16.0.1",
        "http://192.168.0.1",
        "file:///etc/passwd",
        "ftp://trusted.test",
        "https://trusted.test@127.0.0.1",
        "https://trusted.test:8443",
        "https://trusted.test/#fragment",
        "https://trusted.test\\@evil.test",
    ],
)
def test_reject_unsafe_origins(url):
    with pytest.raises(AcquisitionError):
        validate_url(url, {"https://trusted.test"})


@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",
        "169.254.169.254",
        "10.1.2.3",
        "172.31.0.1",
        "192.168.0.1",
        "::1",
        "fc00::1",
        "fe80::1",
        "::ffff:127.0.0.1",
    ],
)
def test_dns_private_and_ipv6_blocked(monkeypatch, address):
    monkeypatch.setattr("socket.getaddrinfo", lambda *a, **k: [(2, 1, 6, "", (address, 443))])
    with pytest.raises(AcquisitionError, match="Non-public"):
        validate_url("https://trusted.test/data", {"https://trusted.test"})


@pytest.mark.parametrize(
    "status,mime,encoding,length,data",
    [
        (302, "application/json", "identity", "2", b"{}"),
        (200, "text/html", "identity", "2", b"{}"),
        (200, "application/json", "gzip", "2", b"{}"),
        (200, "application/json", "identity", "9999", b"{}"),
        (200, "application/json", "identity", "3", b"{}"),
        (200, "application/json", "identity", None, b"x" * 101),
    ],
)
def test_response_security(monkeypatch, status, mime, encoding, length, data):
    class Response:
        def getheader(self, name, default=None):
            return {"Content-Type": mime, "Content-Encoding": encoding, "Content-Length": length}.get(name, default)

        def read(self, size):
            return data[:size]

    response = Response()
    response.status = status

    class Connection:
        def __init__(self, *args):
            pass

        def request(self, *args, **kwargs):
            pass

        def getresponse(self):
            return response

        def close(self):
            pass

    monkeypatch.setattr("backend.safe_http.validate_url", lambda *a: ("trusted.test", ["8.8.8.8"]))
    monkeypatch.setattr("backend.safe_http._PinnedHTTPSConnection", Connection)
    with pytest.raises(AcquisitionError):
        SafeHTTP({"https://trusted.test"}, max_bytes=100).request("https://trusted.test/data")


class MockFHIR:
    allowed_origins = {"https://ehr.test"}

    def __init__(self, raw=None):
        self.prefetched = prefetch(raw or bundle())
        self.calls = []

    def json(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if "/Patient/" in url:
            return copy.deepcopy(self.prefetched["patient"])
        return copy.deepcopy(self.prefetched[url.split("/")[-1].split("?")[0]])


def test_fhir_fetch_and_bearer():
    http = MockFHIR()
    result = FHIRClient("https://ehr.test/fhir", "synthetic-token", http).bundle("SYN-PAT-002")
    assert parse_bundle(result)[0] == parse_bundle(bundle())[0]
    assert len(http.calls) == 7
    assert all(c[1]["headers"]["Authorization"] == "Bearer synthetic-token" for c in http.calls)


def test_fhir_pagination_collects_complete_searchset():
    http = MockFHIR()
    original = http.json

    def paginated(url, **kwargs):
        result = original(url, **kwargs)
        if "Condition?" in url and "page=2" not in url:
            return {
                "resourceType": "Bundle",
                "type": "searchset",
                "entry": [],
                "link": [{"relation": "next", "url": "https://ehr.test/fhir/Condition?page=2"}],
            }
        return result

    http.json = paginated
    assert (
        parse_bundle(FHIRClient("https://ehr.test/fhir", "synthetic-token", http).bundle("SYN-PAT-002"))[0]
        == parse_bundle(bundle())[0]
    )


@pytest.mark.parametrize(
    "failure", ["401", "malformed", "wrong_patient", "unsupported", "pagination_escape", "pagination_loop"]
)
def test_fhir_fetch_fail_closed(failure):
    http = MockFHIR()
    original = http.json

    def bad(url, **kwargs):
        if failure == "401":
            raise AcquisitionError("Upstream response rejected (401)")
        result = original(url, **kwargs)
        if failure == "malformed":
            return {"resourceType": "OperationOutcome"}
        if failure == "wrong_patient" and "/Patient/" in url:
            result["id"] = "SYN-PAT-wrong"
        if "Condition?" in url:
            if failure == "unsupported":
                result["entry"] = [{"resource": {"resourceType": "Device"}}]
            if failure.startswith("pagination"):
                result["link"] = [
                    {"relation": "next", "url": "https://evil.test/Condition" if failure.endswith("escape") else url}
                ]
        return result

    http.json = bad
    with pytest.raises(ValueError):
        FHIRClient("https://ehr.test/fhir", "synthetic-token", http).bundle("SYN-PAT-002")


def test_smart_discovery_and_pkce(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "smart_issuers", ["https://ehr.test/fhir"])

    class HTTP:
        allowed_origins = {"https://ehr.test"}

        def json(self, url):
            return {
                "authorization_endpoint": "https://ehr.test/authorize",
                "token_endpoint": "https://ehr.test/token",
                "code_challenge_methods_supported": ["S256"],
            }

    assert discover("https://ehr.test/fhir", HTTP())["token_endpoint"].endswith("/token")
    assert challenge("dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk") == "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"
    with pytest.raises(AcquisitionError):
        discover("https://evil.test/fhir", HTTP())
    HTTP.allowed_origins = set()
    with pytest.raises(AcquisitionError):
        discover("https://ehr.test/fhir", HTTP())


def test_cds_prefetch_and_fallback(monkeypatch):
    assert parse_bundle(hook_bundle(HookRequest.model_validate(hook())))[0] == parse_bundle(bundle())[0]
    monkeypatch.setattr(get_settings(), "smart_issuers", ["https://ehr.test/fhir"])
    request = hook()
    request.pop("prefetch")
    request.update(
        fhirServer="https://ehr.test/fhir",
        fhirAuthorization={
            "access_token": "synthetic",
            "token_type": "Bearer",
            "expires_in": 60,
            "scope": "patient/Patient.r",
            "subject": "test",
        },
    )
    assert parse_bundle(hook_bundle(HookRequest.model_validate(request), MockFHIR()))[0] == parse_bundle(bundle())[0]


@pytest.mark.parametrize("number,expected", [(2, "potential care gap"), (1, None), (4, "insufficient data"), (5, None)])
def test_cds_cards_use_rule_engine(tmp_path, monkeypatch, number, expected):
    monkeypatch.setattr(get_settings(), "cds_synthetic_fixture_coverage", True)
    context = parse_bundle(bundle(number))[0]
    repo = SQLitePatientRepository(tmp_path / "cards.db")
    patient_id = repo.import_context(context)[0]
    review = repo.create_review(patient_id)
    findings = ClinicalReviewEngine(repo).evaluate(review.review_id, evaluation_context(context))
    result = cards(findings)
    if expected:
        assert any(expected in c["summary"] for c in result)
    else:
        assert result == []
    assert all("suggestions" not in c and "actions" not in c for c in result)


def test_cds_cannot_infer_coverage(monkeypatch):
    monkeypatch.setattr(get_settings(), "cds_synthetic_fixture_coverage", False)
    assert evaluation_context(parse_bundle(bundle())[0]).record_coverage is None


def test_cds_discovery_validation_and_endpoint(tmp_path, monkeypatch):
    import api_server

    repo = SQLitePatientRepository(tmp_path / "cds.db")
    api_server.app.dependency_overrides[get_repository] = lambda: repo
    monkeypatch.setattr(get_settings(), "cds_synthetic_fixture_coverage", True)
    try:
        with TestClient(api_server.app) as client:
            assert client.get("/cds-services").json()["services"][0]["hook"] == "patient-view"
            response = client.post("/cds-services/medical-patient-review", json=hook())
            assert response.status_code == 200, response.text
            assert "potential care gap" in response.json()["cards"][0]["summary"]
            for field in ("patientId", "userId"):
                invalid = hook()
                del invalid["context"][field]
                assert client.post("/cds-services/medical-patient-review", json=invalid).status_code == 422
            invalid = hook()
            invalid["hook"] = "order-sign"
            assert client.post("/cds-services/medical-patient-review", json=invalid).status_code == 422
            assert client.post("/cds-services/unknown", json=hook()).status_code == 404
    finally:
        api_server.app.dependency_overrides.clear()


def test_official_abdm_profiles_and_missing_fields():
    validate_recorded_example()
    assert len(profiles()) == 8
    result = analyze({"resourceType": "Patient", "id": "SYN-PAT-test"}, "Patient")
    assert result["version"] == "6.5.0"
    assert result["missing_required_direct_fields"]
    assert "FHIRPath invariants" in result["unvalidated"]
    for entry in bundle()["entry"]:
        resource = entry["resource"]
        assert analyze(resource, resource["resourceType"])["profile"].startswith("https://nrces.in/")


def test_cds_authorization_and_bounded_body(monkeypatch):
    import api_server
    from backend.security import ActorContext

    monkeypatch.setattr(
        api_server,
        "actor_from_request",
        lambda request: ActorContext("synthetic-auditor", frozenset({"auditor"}), "test", True),
    )
    with TestClient(api_server.app) as client:
        assert client.post("/cds-services/medical-patient-review", json=hook()).status_code == 403
        monkeypatch.setattr(api_server.settings, "max_fhir_body_bytes", 1024)
        assert client.post("/cds-services/medical-patient-review", content=b"x" * 1025).status_code == 413
