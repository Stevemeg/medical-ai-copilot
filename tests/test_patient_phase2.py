"""FHIR boundary, deterministic context, persistence, and API behavior."""

from __future__ import annotations

import copy
import json
import logging
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.fhir_adapter import FHIRInputError, parse_bundle
from backend.patient_api import get_repository
from backend.patient_context import (
    active_conditions,
    active_medications,
    context_hash,
    data_availability,
    known_allergies,
    latest_observation,
    medication_by_code,
    allergy_by_code,
    observations_by_code,
    observations_within,
    timeline,
)
from backend.patient_store import PatientNotFound, ReviewNotFound, SQLitePatientRepository

FIXTURES = Path(__file__).resolve().parents[1] / "data" / "synthetic_fhir"


def bundle(number: int = 3) -> dict:
    return json.loads((FIXTURES / f"syn_pat_{number:03d}.json").read_text(encoding="utf-8"))


def resource(raw: dict, kind: str) -> dict:
    return next(entry["resource"] for entry in raw["entry"] if entry["resource"]["resourceType"] == kind)


def expect_error(raw: dict, code: str) -> None:
    with pytest.raises(FHIRInputError) as caught:
        parse_bundle(raw)
    assert caught.value.code == code
    assert caught.value.path


@pytest.mark.parametrize(
    "number,counts",
    [
        (1, {"Patient": 1, "Condition": 1, "MedicationRequest": 1, "Observation": 1, "Encounter": 1}),
        (2, {"Patient": 1, "Condition": 1, "MedicationRequest": 1, "Observation": 2, "Encounter": 1}),
        (3, {"Patient": 1, "Condition": 2, "MedicationRequest": 2, "Observation": 3, "Encounter": 2}),
        (4, {"Patient": 1, "Condition": 1}),
        (5, {"Patient": 1, "MedicationRequest": 2, "AllergyIntolerance": 1, "Encounter": 1, "Procedure": 1}),
    ],
)
def test_all_synthetic_fixtures(number, counts):
    context, actual = parse_bundle(bundle(number))
    assert actual == counts
    assert context.patient.synthetic_label.startswith("Synthetic Patient")
    assert context.patient.source_patient_id.startswith("SYN-PAT-")
    assert context.source_metadata.synthetic


def test_normalization_reordering_and_transport_format_do_not_change_hash():
    raw = bundle()
    first, _ = parse_bundle(raw)
    changed_order = copy.deepcopy(raw)
    changed_order["entry"].reverse()
    changed_order["entry"][0]["resource"]["meta"] = {"lastUpdated": "2026-01-01T00:00:00Z"}
    second, _ = parse_bundle(changed_order)
    assert first == second
    assert context_hash(first) == context_hash(second)
    next(entry["resource"] for entry in changed_order["entry"] if entry["resource"].get("id") == "obs-c1")[
        "valueQuantity"
    ]["value"] = 8.1
    assert context_hash(parse_bundle(changed_order)[0]) != context_hash(first)


def test_code_display_and_blood_pressure_components():
    context, _ = parse_bundle(bundle(2))
    assert context.conditions[0].code.coding[0].code == "I10"
    bp = latest_observation(context, "http://loinc.org", "85354-9")
    assert bp is not None and len(bp.components) == 2 and bp.value is None
    assert {c.code.coding[0].code for c in bp.components} == {"8480-6", "8462-4"}
    assert bp.effective == "2025-08-09T04:45:00Z"


@pytest.mark.parametrize(
    "mutation,code",
    [
        (lambda b: b.pop("resourceType"), "unsupported_resource"),
        (lambda b: b.update({"entry": {}}), "invalid_fhir"),
        (lambda b: b.update({"entry": []}), "invalid_fhir"),
        (
            lambda b: b["entry"].append({"resource": {"resourceType": "DiagnosticReport", "id": "x"}}),
            "unsupported_resource",
        ),
        (lambda b: b["entry"].append(copy.deepcopy(b["entry"][0])), "duplicate_resource"),
        (lambda b: b["entry"].append({"resource": {"resourceType": "Patient", "id": "other"}}), "invalid_fhir"),
        (lambda b: resource(b, "Observation").update({"effectiveDateTime": "2025-99-99"}), "invalid_fhir"),
        (lambda b: resource(b, "Observation")["valueQuantity"].update({"value": "bad"}), "malformed_observation"),
        (lambda b: resource(b, "Observation")["subject"].update({"reference": "Patient/other"}), "invalid_reference"),
        (
            lambda b: resource(b, "Observation")["subject"].update(
                {"reference": "https://example.org/Patient/SYN-PAT-001"}
            ),
            "invalid_reference",
        ),
        (lambda b: resource(b, "Patient").pop("meta"), "synthetic_only"),
        (
            lambda b: resource(b, "Patient")["identifier"][0].update({"value": "SYN-PAT-999"}),
            "invalid_fhir",
        ),
        (
            lambda b: resource(b, "Patient").update({"birthDate": "1972-04-12T00:00:00Z"}),
            "invalid_fhir",
        ),
    ],
)
def test_invalid_bundle_rejected(mutation, code):
    raw = bundle(1)
    mutation(raw)
    expect_error(raw, code)


def test_missing_resource_type_and_malformed_entry():
    raw = bundle(1)
    resource(raw, "Condition").pop("resourceType")
    expect_error(raw, "invalid_fhir")
    raw = bundle(1)
    raw["entry"].append({"resource": "not an object"})
    expect_error(raw, "invalid_fhir")


def test_dates_status_and_observation_helpers():
    raw = bundle(2)
    raw["entry"].append(
        {
            "resource": {
                "resourceType": "Observation",
                "id": "undated",
                "subject": {"reference": "Patient/SYN-PAT-002"},
                "code": {"coding": [{"system": "http://loinc.org", "code": "85354-9"}]},
                "status": "entered-in-error",
                "valueQuantity": {"value": 1},
            }
        }
    )
    context, _ = parse_bundle(raw)
    assert len(observations_by_code(context, "http://loinc.org", "85354-9")) == 2
    assert latest_observation(context, "http://loinc.org", "85354-9").observation_id == "obs-b2"
    from datetime import date

    assert len(observations_within(context, "http://loinc.org", "85354-9", date(2025, 8, 1), date(2025, 8, 31))) == 1
    assert all(e.source_resource_id != "undated" for e in timeline(context))
    raw = bundle(1)
    resource(raw, "Observation")["effectiveDateTime"] = "2025-08-12T10:00:00"
    expect_error(raw, "invalid_fhir")


def test_context_status_and_availability():
    raw = bundle(5)
    context, _ = parse_bundle(raw)
    assert [m.medication_id for m in active_medications(context)] == ["med-e2"]
    assert len(known_allergies(context)) == 1
    assert "observations" in data_availability(context).missing_data_types
    incomplete, _ = parse_bundle(bundle(4))
    assert len(active_conditions(incomplete)) == 1
    assert "allergies" in data_availability(incomplete).missing_data_types
    assert "overdue" not in json.dumps(data_availability(incomplete).model_dump())
    inactive = bundle(4)
    resource(inactive, "Condition")["clinicalStatus"]["coding"][0]["code"] = "inactive"
    assert active_conditions(parse_bundle(inactive)[0]) == ()
    assert data_availability(parse_bundle(bundle(1))[0]).last_observation_dates["Hemoglobin A1c"] == "2025-08-12"
    raw = bundle(5)
    resource(raw, "MedicationRequest")["medicationCodeableConcept"]["coding"] = [
        {"system": "urn:synthetic-medication", "code": "MED-1", "display": "Synthetic code"}
    ]
    coded = parse_bundle(raw)[0]
    assert not medication_by_code(coded, "urn:synthetic-medication", "MED-1")
    assert len(allergy_by_code(coded, "http://snomed.info/sct", "91936005")) == 1
    next(entry["resource"] for entry in raw["entry"] if entry["resource"].get("id") == "med-e2")[
        "medicationCodeableConcept"
    ]["coding"] = [{"system": "urn:synthetic-medication", "code": "MED-2"}]
    active_coded = parse_bundle(raw)[0]
    assert len(medication_by_code(active_coded, "urn:synthetic-medication", "MED-2")) == 1


def test_timeline_order_undated_and_ties():
    raw = bundle(1)
    resource(raw, "Observation").pop("effectiveDateTime")
    raw["entry"].append(
        {
            "resource": {
                "resourceType": "Procedure",
                "id": "same-time",
                "subject": {"reference": "Patient/SYN-PAT-001"},
                "code": {"text": "Synthetic procedure"},
                "status": "completed",
                "performedDateTime": "2025-08-12T09:00:00Z",
            }
        }
    )
    context, _ = parse_bundle(raw)
    events = timeline(context)
    assert events[-1].event_id == "observation:obs-a1"
    assert events == timeline(context)
    assert events[0].timestamp == "2025-08-12T09:00:00Z"
    assert [event.event_type for event in events[:2]] == ["encounter", "procedure"]


def test_sqlite_idempotent_update_and_immutable_review(tmp_path):
    path = tmp_path / "patients.db"
    repo = SQLitePatientRepository(path)
    context = parse_bundle(bundle(3))[0]
    patient_id, digest, changed = repo.import_context(context)
    assert changed
    assert repo.import_context(context) == (patient_id, digest, False)
    review = repo.create_review(patient_id)
    assert review.patient_context_hash == digest
    assert not review.findings and not review.evidence
    changed_raw = bundle(3)
    resource(changed_raw, "Observation")["valueQuantity"]["value"] = 8.1
    new_digest = repo.import_context(parse_bundle(changed_raw)[0])[1]
    assert new_digest != digest
    fresh = SQLitePatientRepository(path)
    assert fresh.get_review(review.review_id).patient_context_hash == digest
    assert [item.review_id for item in fresh.list_reviews(patient_id)] == [review.review_id]
    assert fresh.create_review(patient_id).review_id != review.review_id
    assert context_hash(fresh.get_review(review.review_id).patient_snapshot) == digest
    assert fresh.get_patient(patient_id)[1] == new_digest
    completed = fresh.complete_review(review.review_id)
    assert completed.status.value == "completed"
    assert fresh.complete_review(review.review_id) == completed
    assert len(fresh.list_patients()) == 1
    with pytest.raises(PatientNotFound):
        fresh.get_patient("missing")
    with pytest.raises(ReviewNotFound):
        fresh.get_review("missing")


def test_corrupted_patient_snapshot_is_rejected(tmp_path):
    path = tmp_path / "patients.db"
    repo = SQLitePatientRepository(path)
    patient_id = repo.import_context(parse_bundle(bundle(1))[0])[0]
    with sqlite3.connect(path) as db:
        db.execute("UPDATE patients SET context_hash='incorrect' WHERE patient_id=?", (patient_id,))
    with pytest.raises(ValueError, match="hash mismatch"):
        repo.get_patient(patient_id)
    with pytest.raises(ValueError, match="hash mismatch"):
        repo.list_patients()


@pytest.fixture
def client(tmp_path):
    from api_server import app

    repo = SQLitePatientRepository(tmp_path / "api.db")
    app.dependency_overrides[get_repository] = lambda: repo
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_api_end_to_end_and_ask_evidence(client, monkeypatch):
    raw = bundle(2)
    valid = client.post("/v1/fhir/validate", json=raw)
    assert valid.status_code == 200 and valid.json()["valid"]
    assert valid.json()["resource_counts"]["Observation"] == 2
    assert client.get("/v1/patients").json() == []
    imported = client.post("/v1/patients/import", json=raw)
    assert imported.status_code == 200
    patient_id = imported.json()["patient_id"]
    assert client.post("/v1/patients/import", json=raw).json()["changed"] is False
    assert len(client.get("/v1/patients").json()) == 1
    patient = client.get(f"/v1/patients/{patient_id}")
    assert patient.status_code == 200
    assert patient.json()["context_hash"] == imported.json()["context_hash"]
    assert patient.json()["timeline"] and patient.json()["data_availability"]
    created = client.post(f"/v1/patients/{patient_id}/reviews")
    assert created.status_code == 201
    review_id = created.json()["review_id"]
    assert [item["review_id"] for item in client.get(f"/v1/patients/{patient_id}/reviews").json()] == [review_id]
    assert client.get(f"/v1/reviews/{review_id}").json()["patient_context_hash"] == imported.json()["context_hash"]
    assert client.post(f"/v1/reviews/{review_id}/complete").json()["status"] == "completed"
    assert client.get("/v1/knowledge-sources").status_code == 200
    assert client.get("/api/sources").status_code == 200
    import api_server

    monkeypatch.setattr(
        api_server,
        "answer_question",
        lambda question, policy: {"answer": "No matching evidence.", "sources": [], "citations": []},
    )
    ask = client.post("/api/ask", json={"question": "Unrelated question"})
    assert ask.status_code == 200 and ask.json()["answer"] == "No matching evidence."


def test_api_validation_errors_and_not_found(client):
    bad = bundle(1)
    resource(bad, "Observation")["valueQuantity"]["value"] = "secret-marker"
    result = client.post("/v1/fhir/validate", json=bad)
    assert result.status_code == 200 and not result.json()["valid"]
    assert result.json()["errors"][0]["code"] == "malformed_observation"
    assert client.post("/v1/patients/import", json=bad).status_code == 422
    assert client.get("/v1/patients").json() == []
    assert client.get("/v1/patients/missing").json()["detail"]["code"] == "patient_not_found"
    assert client.post("/v1/patients/missing/reviews").status_code == 404
    assert client.get("/v1/reviews/missing").json()["detail"]["code"] == "review_not_found"
    assert client.post("/v1/demo-patients/syn_pat_004/load").status_code == 200


def test_patient_payload_not_logged(client, caplog):
    raw = bundle(1)
    raw["entry"][0]["resource"]["name"][0]["text"] = "Synthetic Patient SECRET_MARKER"
    with caplog.at_level(logging.INFO):
        response = client.post("/v1/patients/import", json=raw)
    assert response.status_code == 200
    assert "SECRET_MARKER" not in caplog.text


def test_api_corrupted_snapshot_returns_structured_storage_error(client):
    imported = client.post("/v1/patients/import", json=bundle(1)).json()
    repo = client.app.dependency_overrides[get_repository]()
    with sqlite3.connect(repo.path) as db:
        db.execute("UPDATE patients SET context_hash='incorrect' WHERE patient_id=?", (imported["patient_id"],))
    response = client.get(f"/v1/patients/{imported['patient_id']}")
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "storage_error"
