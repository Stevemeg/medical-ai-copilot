"""Strict adapter for a documented, synthetic FHIR R4-compatible Bundle subset.

This is structural validation of fields we consume, not full FHIR conformance.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from datetime import date, datetime, timezone
from typing import Any, Mapping

from backend.patient_models import (
    Allergy,
    Coding,
    Concept,
    Condition,
    Encounter,
    Medication,
    Observation,
    ObservationComponent,
    Patient,
    PatientContext,
    Procedure,
    Quantity,
)

SUPPORTED_TYPES = frozenset(
    {"Patient", "Condition", "Observation", "MedicationRequest", "AllergyIntolerance", "Encounter", "Procedure"}
)
FHIR_ID = re.compile(r"^[A-Za-z0-9.-]{1,64}$")
SYNTHETIC_SYSTEM = "urn:medical-ai-copilot:synthetic"


class FHIRInputError(ValueError):
    def __init__(self, code: str, path: str, message: str):
        self.code, self.path, self.message = code, path, message
        super().__init__(message)

    def as_dict(self) -> dict[str, str]:
        return {"code": self.code, "path": self.path, "message": self.message}


def _fail(code: str, path: str, message: str) -> None:
    raise FHIRInputError(code, path, message)


def _obj(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        _fail("invalid_fhir", path, "Expected an object")
    return value


def _list(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        _fail("invalid_fhir", path, "Expected an array")
    return value


def _string(value: Any, path: str, required: bool = False) -> str | None:
    if value is None and not required:
        return None
    if not isinstance(value, str) or not value.strip():
        _fail("invalid_fhir", path, "Expected a nonempty string")
    return value.strip()


def _date(value: Any, path: str) -> str | None:
    raw = _string(value, path)
    if raw is None:
        return None
    try:
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw):
            return date.fromisoformat(raw).isoformat()
        if "T" not in raw:
            raise ValueError("Unsupported date precision")
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("A time requires a timezone offset")
        return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    except ValueError:
        _fail("invalid_fhir", path, "Invalid date or dateTime")
    return None


def _concept(value: Any, path: str) -> Concept:
    obj = _obj(value, path)
    codings: list[Coding] = []
    for index, item in enumerate(_list(obj.get("coding", []), f"{path}.coding")):
        item = _obj(item, f"{path}.coding[{index}]")
        coding = Coding(
            system=_string(item.get("system"), f"{path}.coding[{index}].system"),
            code=_string(item.get("code"), f"{path}.coding[{index}].code"),
            display=_string(item.get("display"), f"{path}.coding[{index}].display"),
        )
        if not coding.code and not coding.display:
            _fail("invalid_fhir", f"{path}.coding[{index}]", "Coding requires code or display")
        codings.append(coding)
    text = _string(obj.get("text"), f"{path}.text")
    if not text and not codings:
        _fail("invalid_fhir", path, "CodeableConcept requires text or coding")
    return Concept(
        text=text, coding=tuple(sorted(codings, key=lambda c: (c.system or "", c.code or "", c.display or "")))
    )


def _status(value: Any, path: str) -> str | None:
    if value is None:
        return None
    obj = _obj(value, path)
    entries = _list(obj.get("coding", []), f"{path}.coding")
    if not entries:
        _fail("invalid_fhir", path, "Status requires a coding")
    return _string(_obj(entries[0], f"{path}.coding[0]").get("code"), f"{path}.coding[0].code", True)


def _quantity(value: Any, path: str) -> Quantity:
    obj = _obj(value, path)
    number = obj.get("value")
    if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number):
        _fail("malformed_observation", f"{path}.value", "Quantity requires a finite numeric value")
    assert isinstance(number, (int, float))
    return Quantity(
        value=float(number),
        unit=_string(obj.get("unit"), f"{path}.unit"),
        system=_string(obj.get("system"), f"{path}.system"),
        code=_string(obj.get("code"), f"{path}.code"),
    )


def _patient_reference(resource: Mapping[str, Any], kind: str, patient_id: str) -> None:
    field = "patient" if kind == "AllergyIntolerance" else "subject"
    ref = _string(_obj(resource.get(field), f"{kind}.{field}").get("reference"), f"{kind}.{field}.reference", True)
    if not re.fullmatch(r"Patient/[A-Za-z0-9.-]{1,64}", ref or ""):
        _fail("invalid_reference", f"{kind}.{field}.reference", "Only relative Patient/<id> references are supported")
    if ref != f"Patient/{patient_id}":
        _fail("invalid_reference", f"{kind}.{field}.reference", "Referenced patient is not in this Bundle")


def _patient(resource: Mapping[str, Any]) -> Patient:
    source_id = _string(resource.get("id"), "Patient.id", True)
    identifiers = _list(resource.get("identifier", []), "Patient.identifier")
    synthetic_ids = [
        _string(_obj(item, "Patient.identifier[]").get("value"), "Patient.identifier[].value")
        for item in identifiers
        if _obj(item, "Patient.identifier[]").get("system") == SYNTHETIC_SYSTEM
    ]
    tags = _list(_obj(resource.get("meta", {}), "Patient.meta").get("tag", []), "Patient.meta.tag")
    marked = any(
        _obj(tag, "Patient.meta.tag[]").get("code") == "synthetic"
        and _obj(tag, "Patient.meta.tag[]").get("system") == SYNTHETIC_SYSTEM
        for tag in tags
    )
    if not marked or len(synthetic_ids) != 1 or not (synthetic_ids[0] or "").startswith("SYN-PAT-"):
        _fail("synthetic_only", "Patient", "Import requires a synthetic tag and SYN-PAT identifier")
    if synthetic_ids[0] != source_id:
        _fail("invalid_fhir", "Patient.identifier", "Synthetic identifier must match Patient.id")
    names = _list(resource.get("name", []), "Patient.name")
    label = _string(_obj(names[0], "Patient.name[0]").get("text"), "Patient.name[0].text", True) if names else None
    if not label or not label.startswith("Synthetic Patient"):
        _fail("synthetic_only", "Patient.name", "Use an explicit Synthetic Patient display label")
    assert label is not None
    gender = _string(resource.get("gender"), "Patient.gender")
    if gender not in (None, "male", "female", "other", "unknown"):
        _fail("invalid_fhir", "Patient.gender", "Unsupported gender code")
    birth_date = _date(resource.get("birthDate"), "Patient.birthDate")
    if birth_date and len(birth_date) != 10:
        _fail("invalid_fhir", "Patient.birthDate", "Patient birthDate must be a date")
    return Patient(
        source_patient_id=source_id or "",
        synthetic_label=label,
        birth_date=birth_date,
        gender=gender,
    )


def parse_bundle(raw: Any) -> tuple[PatientContext, dict[str, int]]:
    """Validate and normalize one synthetic patient collection Bundle."""
    bundle = _obj(raw, "Bundle")
    if bundle.get("resourceType") != "Bundle":
        _fail("unsupported_resource", "resourceType", "Expected a Bundle")
    if bundle.get("type") != "collection":
        _fail("unsupported_resource", "Bundle.type", "Only collection Bundles are supported")
    entries = _list(bundle.get("entry"), "Bundle.entry")
    if not entries:
        _fail("invalid_fhir", "Bundle.entry", "Bundle must contain one Patient")
    resources: list[tuple[str, Mapping[str, Any]]] = []
    seen: set[tuple[str, str]] = set()
    counts: Counter[str] = Counter()
    for index, entry in enumerate(entries):
        item = _obj(entry, f"Bundle.entry[{index}]")
        resource = _obj(item.get("resource"), f"Bundle.entry[{index}].resource")
        kind = _string(resource.get("resourceType"), f"Bundle.entry[{index}].resource.resourceType", True) or ""
        if kind not in SUPPORTED_TYPES:
            _fail(
                "unsupported_resource",
                f"Bundle.entry[{index}].resource.resourceType",
                "Resource type is outside the supported subset",
            )
        resource_id = _string(resource.get("id"), f"Bundle.entry[{index}].resource.id", True) or ""
        if not FHIR_ID.fullmatch(resource_id):
            _fail("invalid_fhir", f"Bundle.entry[{index}].resource.id", "Invalid FHIR id")
        key = kind, resource_id
        if key in seen:
            _fail("duplicate_resource", f"Bundle.entry[{index}].resource.id", "Duplicate resource type and id")
        seen.add(key)
        counts[kind] += 1
        resources.append((kind, resource))
    patients = [resource for kind, resource in resources if kind == "Patient"]
    if len(patients) != 1:
        _fail("invalid_fhir", "Bundle.entry", "Exactly one Patient is required")
    patient = _patient(patients[0])
    output: dict[str, list[Any]] = {
        name: [] for name in ("conditions", "observations", "medications", "allergies", "encounters", "procedures")
    }
    for kind, resource in resources:
        if kind == "Patient":
            continue
        _patient_reference(resource, kind, patient.source_patient_id)
        rid = str(resource["id"])
        if kind == "Condition":
            output["conditions"].append(
                Condition(
                    condition_id=rid,
                    code=_concept(resource.get("code"), f"Condition/{rid}.code"),
                    clinical_status=_status(resource.get("clinicalStatus"), f"Condition/{rid}.clinicalStatus"),
                    onset=_date(resource.get("onsetDateTime") or resource.get("onsetDate"), f"Condition/{rid}.onset"),
                    recorded_date=_date(resource.get("recordedDate"), f"Condition/{rid}.recordedDate"),
                )
            )
        elif kind == "Observation":
            components = []
            for index, component in enumerate(_list(resource.get("component", []), f"Observation/{rid}.component")):
                component = _obj(component, f"Observation/{rid}.component[{index}]")
                components.append(
                    ObservationComponent(
                        code=_concept(component.get("code"), f"Observation/{rid}.component[{index}].code"),
                        value=_quantity(
                            component.get("valueQuantity"), f"Observation/{rid}.component[{index}].valueQuantity"
                        ),
                    )
                )
            value = (
                _quantity(resource["valueQuantity"], f"Observation/{rid}.valueQuantity")
                if "valueQuantity" in resource
                else None
            )
            if value is None and not components:
                _fail(
                    "malformed_observation", f"Observation/{rid}", "A numeric value or numeric components are required"
                )
            output["observations"].append(
                Observation(
                    observation_id=rid,
                    code=_concept(resource.get("code"), f"Observation/{rid}.code"),
                    status=_string(resource.get("status"), f"Observation/{rid}.status", True) or "",
                    effective=_date(resource.get("effectiveDateTime"), f"Observation/{rid}.effectiveDateTime"),
                    value=value,
                    components=tuple(sorted(components, key=lambda c: c.code.model_dump_json())),
                )
            )
        elif kind == "MedicationRequest":
            instructions = _list(resource.get("dosageInstruction", []), f"MedicationRequest/{rid}.dosageInstruction")
            dosage = (
                _string(_obj(instructions[0], "dosageInstruction[0]").get("text"), "dosageInstruction[0].text")
                if instructions
                else None
            )
            output["medications"].append(
                Medication(
                    medication_id=rid,
                    code=_concept(
                        resource.get("medicationCodeableConcept"), f"MedicationRequest/{rid}.medicationCodeableConcept"
                    ),
                    status=_string(resource.get("status"), f"MedicationRequest/{rid}.status", True) or "",
                    intent=_string(resource.get("intent"), f"MedicationRequest/{rid}.intent", True) or "",
                    authored_on=_date(resource.get("authoredOn"), f"MedicationRequest/{rid}.authoredOn"),
                    dosage_text=dosage,
                )
            )
        elif kind == "AllergyIntolerance":
            categories = tuple(
                sorted(
                    _string(x, f"AllergyIntolerance/{rid}.category[]", True) or ""
                    for x in _list(resource.get("category", []), f"AllergyIntolerance/{rid}.category")
                )
            )
            reactions = []
            for reaction in _list(resource.get("reaction", []), f"AllergyIntolerance/{rid}.reaction"):
                reaction = _obj(reaction, f"AllergyIntolerance/{rid}.reaction[]")
                for manifestation in _list(
                    reaction.get("manifestation", []), f"AllergyIntolerance/{rid}.reaction[].manifestation"
                ):
                    reactions.append(
                        _concept(manifestation, f"AllergyIntolerance/{rid}.reaction[].manifestation[]").display
                    )
            output["allergies"].append(
                Allergy(
                    allergy_id=rid,
                    code=_concept(resource.get("code"), f"AllergyIntolerance/{rid}.code"),
                    clinical_status=_status(resource.get("clinicalStatus"), f"AllergyIntolerance/{rid}.clinicalStatus"),
                    verification_status=_status(
                        resource.get("verificationStatus"), f"AllergyIntolerance/{rid}.verificationStatus"
                    ),
                    category=categories,
                    criticality=_string(resource.get("criticality"), f"AllergyIntolerance/{rid}.criticality"),
                    recorded_date=_date(resource.get("recordedDate"), f"AllergyIntolerance/{rid}.recordedDate"),
                    reactions=tuple(sorted(reactions)),
                )
            )
        elif kind == "Encounter":
            period = _obj(resource.get("period", {}), f"Encounter/{rid}.period")
            start = _date(period.get("start"), f"Encounter/{rid}.period.start")
            end = _date(period.get("end"), f"Encounter/{rid}.period.end")
            if start and end and start > end:
                _fail("invalid_fhir", f"Encounter/{rid}.period", "End precedes start")
            types = _list(resource.get("type", []), f"Encounter/{rid}.type")
            output["encounters"].append(
                Encounter(
                    encounter_id=rid,
                    status=_string(resource.get("status"), f"Encounter/{rid}.status", True) or "",
                    class_code=_string(
                        _obj(resource.get("class", {}), f"Encounter/{rid}.class").get("code"),
                        f"Encounter/{rid}.class.code",
                    ),
                    type=_concept(types[0], f"Encounter/{rid}.type[0]") if types else None,
                    start=start,
                    end=end,
                )
            )
        elif kind == "Procedure":
            output["procedures"].append(
                Procedure(
                    procedure_id=rid,
                    code=_concept(resource.get("code"), f"Procedure/{rid}.code"),
                    status=_string(resource.get("status"), f"Procedure/{rid}.status", True) or "",
                    performed=_date(
                        resource.get("performedDateTime") or resource.get("performedDate"), f"Procedure/{rid}.performed"
                    ),
                )
            )

    def ordered(name: str, id_field: str) -> tuple[Any, ...]:
        return tuple(sorted(output[name], key=lambda item: getattr(item, id_field)))

    return PatientContext(
        patient=patient,
        conditions=ordered("conditions", "condition_id"),
        observations=ordered("observations", "observation_id"),
        medications=ordered("medications", "medication_id"),
        allergies=ordered("allergies", "allergy_id"),
        encounters=ordered("encounters", "encounter_id"),
        procedures=ordered("procedures", "procedure_id"),
    ), dict(sorted(counts.items()))
