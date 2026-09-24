"""Deterministic queries, provenance hash, timeline, and record availability."""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timezone

from backend.patient_models import (
    Allergy,
    Condition,
    DataAvailability,
    Medication,
    Observation,
    PatientContext,
    TimelineEvent,
)


def context_hash(context: PatientContext) -> str:
    payload = context.model_dump(mode="json")
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def active_conditions(context: PatientContext) -> tuple[Condition, ...]:
    return tuple(c for c in context.conditions if c.clinical_status == "active")


def active_medications(context: PatientContext) -> tuple[Medication, ...]:
    return tuple(m for m in context.medications if m.status == "active")


def medication_by_code(context: PatientContext, system: str, code: str) -> tuple[Medication, ...]:
    return tuple(m for m in active_medications(context) if m.code.has_code(system, code))


def known_allergies(context: PatientContext) -> tuple[Allergy, ...]:
    return tuple(
        a
        for a in context.allergies
        if a.clinical_status != "inactive" and a.verification_status not in ("entered-in-error", "refuted")
    )


def allergy_by_code(context: PatientContext, system: str, code: str) -> tuple[Allergy, ...]:
    return tuple(a for a in known_allergies(context) if a.code.has_code(system, code))


def observations_by_code(context: PatientContext, system: str, code: str) -> tuple[Observation, ...]:
    return tuple(
        o
        for o in context.observations
        if o.status not in ("entered-in-error", "cancelled") and o.code.has_code(system, code)
    )


def _time_key(value: str | None) -> tuple[str, str]:
    if value is None:
        return ("", "")
    if len(value) == 10:
        return (value, "")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    utc = parsed.astimezone(timezone.utc).isoformat()
    return (utc[:10], utc[11:])


def latest_observation(context: PatientContext, system: str, code: str) -> Observation | None:
    """Return the latest *dated* observation; ties use observation ID."""
    matches = (o for o in observations_by_code(context, system, code) if o.effective)
    return max(matches, key=lambda o: (_time_key(o.effective), o.observation_id), default=None)


def observations_with_unknown_date(context: PatientContext, system: str, code: str) -> tuple[Observation, ...]:
    return tuple(o for o in observations_by_code(context, system, code) if not o.effective)


def observations_within(
    context: PatientContext, system: str, code: str, start: date, end: date
) -> tuple[Observation, ...]:
    return tuple(
        o
        for o in observations_by_code(context, system, code)
        if o.effective and start <= date.fromisoformat(o.effective[:10]) <= end
    )


def timeline(context: PatientContext) -> tuple[TimelineEvent, ...]:
    events: list[TimelineEvent] = []
    for c in context.conditions:
        events.append(
            TimelineEvent(
                event_id=f"condition:{c.condition_id}",
                event_type="condition",
                timestamp=c.onset or c.recorded_date,
                title=c.code.display,
                details={"status": c.clinical_status or "unknown"},
                source_resource_id=c.condition_id,
            )
        )
    for o in context.observations:
        if o.status in ("entered-in-error", "cancelled"):
            continue
        details = {"status": o.status}
        if o.value:
            details["value"] = f"{o.value.value:g} {o.value.unit or ''}".strip()
        for component in o.components:
            details[component.code.display] = f"{component.value.value:g} {component.value.unit or ''}".strip()
        events.append(
            TimelineEvent(
                event_id=f"observation:{o.observation_id}",
                event_type="observation",
                timestamp=o.effective,
                title=o.code.display,
                details=details,
                source_resource_id=o.observation_id,
            )
        )
    for m in context.medications:
        events.append(
            TimelineEvent(
                event_id=f"medication:{m.medication_id}",
                event_type="medication",
                timestamp=m.authored_on,
                title=m.code.display,
                details={"status": m.status, "intent": m.intent},
                source_resource_id=m.medication_id,
            )
        )
    for a in context.allergies:
        events.append(
            TimelineEvent(
                event_id=f"allergy:{a.allergy_id}",
                event_type="allergy",
                timestamp=a.recorded_date,
                title=a.code.display,
                details={"status": a.clinical_status or "unknown"},
                source_resource_id=a.allergy_id,
            )
        )
    for e in context.encounters:
        events.append(
            TimelineEvent(
                event_id=f"encounter:{e.encounter_id}",
                event_type="encounter",
                timestamp=e.start,
                title=e.type.display if e.type else "Encounter",
                details={"status": e.status, "class": e.class_code or "unknown"},
                source_resource_id=e.encounter_id,
            )
        )
    for p in context.procedures:
        events.append(
            TimelineEvent(
                event_id=f"procedure:{p.procedure_id}",
                event_type="procedure",
                timestamp=p.performed,
                title=p.code.display,
                details={"status": p.status},
                source_resource_id=p.procedure_id,
            )
        )
    # Dated newest first; undated events follow, ordered by type and source ID.
    events.sort(key=lambda e: (e.event_type, e.source_resource_id))
    events.sort(key=lambda e: _time_key(e.timestamp), reverse=True)
    return tuple(events)


def data_availability(context: PatientContext) -> DataAvailability:
    collections = {
        "conditions": context.conditions,
        "medications": context.medications,
        "allergies": context.allergies,
        "observations": context.observations,
        "encounters": context.encounters,
        "procedures": context.procedures,
    }
    available = tuple(sorted(name for name, values in collections.items() if values))
    missing = tuple(sorted(name for name, values in collections.items() if not values))
    dates: dict[str, str] = {}
    for observation in context.observations:
        if observation.effective:
            key = observation.code.display
            if key not in dates or _time_key(observation.effective) > _time_key(dates[key]):
                dates[key] = observation.effective
    return DataAvailability(
        available_data_types=available, missing_data_types=missing, last_observation_dates=dict(sorted(dates.items()))
    )
