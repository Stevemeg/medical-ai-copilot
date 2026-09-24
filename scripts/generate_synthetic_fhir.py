"""Regenerate the five explicitly fictional FHIR R4 collection fixtures."""

import json
from pathlib import Path

OUTPUT = Path(__file__).resolve().parents[1] / "data" / "synthetic_fhir"
OUTPUT.mkdir(parents=True, exist_ok=True)


def concept(text, system=None, code=None):
    result = {"text": text}
    if system and code:
        result["coding"] = [{"system": system, "code": code, "display": text}]
    return result


def patient(number, birth, gender):
    source_id = f"SYN-PAT-{number:03d}"
    return {
        "resourceType": "Patient",
        "id": source_id,
        "meta": {"tag": [{"system": "urn:medical-ai-copilot:synthetic", "code": "synthetic"}]},
        "identifier": [{"system": "urn:medical-ai-copilot:synthetic", "value": source_id}],
        "name": [{"text": f"Synthetic Patient {chr(64 + number)}"}],
        "birthDate": birth,
        "gender": gender,
    }


def link(kind, rid, number, **fields):
    reference = {"reference": f"Patient/SYN-PAT-{number:03d}"}
    return {
        "resourceType": kind,
        "id": rid,
        "patient" if kind == "AllergyIntolerance" else "subject": reference,
        **fields,
    }


def condition(rid, number, title, code, onset):
    return link(
        "Condition",
        rid,
        number,
        code=concept(title, "http://hl7.org/fhir/sid/icd-10", code),
        clinicalStatus={"coding": [{"code": "active"}]},
        onsetDateTime=onset,
        recordedDate=onset,
    )


def medication(rid, number, title, authored, status="active"):
    return link(
        "MedicationRequest",
        rid,
        number,
        medicationCodeableConcept=concept(title),
        status=status,
        intent="order",
        authoredOn=authored,
    )


def observation(rid, number, title, code, value, unit, effective):
    return link(
        "Observation",
        rid,
        number,
        code=concept(title, "http://loinc.org", code),
        status="final",
        effectiveDateTime=effective,
        valueQuantity={"value": value, "unit": unit},
    )


def blood_pressure(rid, number, systolic, diastolic, effective):
    return link(
        "Observation",
        rid,
        number,
        code=concept("Blood pressure", "http://loinc.org", "85354-9"),
        status="final",
        effectiveDateTime=effective,
        component=[
            {
                "code": concept("Systolic blood pressure", "http://loinc.org", "8480-6"),
                "valueQuantity": {"value": systolic, "unit": "mm[Hg]"},
            },
            {
                "code": concept("Diastolic blood pressure", "http://loinc.org", "8462-4"),
                "valueQuantity": {"value": diastolic, "unit": "mm[Hg]"},
            },
        ],
    )


def encounter(rid, number, start, end):
    return link(
        "Encounter",
        rid,
        number,
        status="finished",
        **{"class": {"code": "AMB"}},
        type=[concept("Outpatient visit")],
        period={"start": start, "end": end},
    )


fixtures = {
    1: [
        patient(1, "1972-04-12", "female"),
        condition("cond-a1", 1, "Type 2 diabetes mellitus", "E11.9", "2021-05-11"),
        medication("med-a1", 1, "Metformin", "2025-04-08"),
        observation("obs-a1", 1, "Hemoglobin A1c", "4548-4", 7.2, "%", "2025-08-12"),
        encounter("enc-a1", 1, "2025-08-12T09:00:00Z", "2025-08-12T09:30:00Z"),
    ],
    2: [
        patient(2, "1965-11-02", "male"),
        condition("cond-b1", 2, "Essential hypertension", "I10", "2019-03-04"),
        medication("med-b1", 2, "Amlodipine", "2025-01-05"),
        blood_pressure("obs-b1", 2, 138, 84, "2025-07-09T10:15:00+05:30"),
        blood_pressure("obs-b2", 2, 132, 82, "2025-08-09T10:15:00+05:30"),
        encounter("enc-b1", 2, "2025-08-09T10:00:00+05:30", "2025-08-09T10:45:00+05:30"),
    ],
    3: [
        patient(3, "1958-02-18", "female"),
        condition("cond-c1", 3, "Type 2 diabetes mellitus", "E11.9", "2020-06-03"),
        condition("cond-c2", 3, "Essential hypertension", "I10", "2022-09-20"),
        medication("med-c1", 3, "Metformin", "2025-01-08"),
        medication("med-c2", 3, "Amlodipine", "2025-01-08"),
        observation("obs-c1", 3, "Hemoglobin A1c", "4548-4", 7.4, "%", "2025-02-12"),
        observation("obs-c2", 3, "Hemoglobin A1c", "4548-4", 7.1, "%", "2025-08-12"),
        blood_pressure("obs-c3", 3, 136, 82, "2025-08-12"),
        encounter("enc-c1", 3, "2025-02-12T11:00:00Z", "2025-02-12T11:25:00Z"),
        encounter("enc-c2", 3, "2025-08-12T11:00:00Z", "2025-08-12T11:25:00Z"),
    ],
    4: [patient(4, "1980-07-27", "other"), condition("cond-d1", 4, "Type 2 diabetes mellitus", "E11.9", "2024-01-10")],
    5: [
        patient(5, "1975-09-30", "female"),
        medication("med-e1", 5, "Amoxicillin", "2024-01-05", status="stopped"),
        medication("med-e2", 5, "Doxycycline", "2025-03-07"),
        link(
            "AllergyIntolerance",
            "allergy-e1",
            5,
            code=concept("Penicillin", "http://snomed.info/sct", "91936005"),
            clinicalStatus={"coding": [{"code": "active"}]},
            verificationStatus={"coding": [{"code": "confirmed"}]},
            category=["medication"],
            criticality="high",
            recordedDate="2023-06-04",
            reaction=[{"manifestation": [concept("Rash")]}],
        ),
        encounter("enc-e1", 5, "2025-03-07T08:00:00Z", "2025-03-07T08:20:00Z"),
        link(
            "Procedure",
            "proc-e1",
            5,
            code=concept("Synthetic specimen collection"),
            status="completed",
            performedDateTime="2025-03-07T08:15:00Z",
        ),
    ],
}

for number, resources in fixtures.items():
    bundle = {
        "resourceType": "Bundle",
        "id": f"synthetic-bundle-{number}",
        "type": "collection",
        "entry": [{"resource": resource} for resource in resources],
    }
    (OUTPUT / f"syn_pat_{number:03d}.json").write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8")
