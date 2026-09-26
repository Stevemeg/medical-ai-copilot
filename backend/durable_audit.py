"""Transactional, keyed audit chain for minimized operational events."""

import hashlib
import hmac
import json
from datetime import timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.db import AuditCheckpoint, AuditEvent, AuditHead, utcnow
from backend.settings import get_settings

GENESIS = "0" * 64
CHECKPOINT_INTERVAL = 100


def canonical(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def signature(value: bytes) -> str:
    return hmac.new(get_settings().audit_hmac_key.get_secret_value().encode("utf-8"), value, hashlib.sha256).hexdigest()


def _fields(event: AuditEvent) -> dict:
    return {
        "sequence_number": event.sequence_number,
        "occurred_at": event.occurred_at.astimezone(timezone.utc).isoformat(),
        "actor_subject": event.actor_subject,
        "actor_roles": event.actor_roles,
        "request_id": event.request_id,
        "event_type": event.event_type,
        "resource_type": event.resource_type,
        "resource_id": event.resource_id,
        "event_payload": event.event_payload,
        "previous_hash": event.previous_hash,
    }


def append_event(
    session: Session,
    *,
    event_type: str,
    resource_type: str,
    resource_id: str,
    payload: dict | None = None,
    actor_subject: str = "system",
    actor_roles: list[str] | None = None,
    request_id: str = "system",
) -> AuditEvent:
    # The single row lock serializes sequence assignment and predecessor selection.
    head = session.execute(select(AuditHead).where(AuditHead.id == 1).with_for_update()).scalar_one()
    event = AuditEvent(
        sequence_number=head.last_sequence + 1,
        occurred_at=utcnow(),
        actor_subject=actor_subject,
        actor_roles=actor_roles or [],
        request_id=request_id,
        event_type=event_type,
        resource_type=resource_type,
        resource_id=resource_id,
        event_payload=payload or {},
        previous_hash=head.last_hash,
        event_hash="",
        event_hmac="",
    )
    event.event_hash = hashlib.sha256(canonical(_fields(event))).hexdigest()
    event.event_hmac = signature(event.event_hash.encode("ascii"))
    session.add(event)
    head.last_sequence, head.last_hash = event.sequence_number, event.event_hash
    if event.sequence_number % CHECKPOINT_INTERVAL == 0:
        checkpoint_payload = canonical({"sequence": event.sequence_number, "hash": event.event_hash})
        session.add(
            AuditCheckpoint(
                last_sequence=event.sequence_number,
                last_event_hash=event.event_hash,
                checkpoint_hmac=signature(checkpoint_payload),
            )
        )
    return event


def verify_chain(session: Session) -> dict:
    expected_hash, expected_seq = GENESIS, 0
    checkpoints = {c.last_sequence: c for c in session.scalars(select(AuditCheckpoint))}
    for event in session.scalars(select(AuditEvent).order_by(AuditEvent.sequence_number)):
        if event.sequence_number != expected_seq + 1 or event.previous_hash != expected_hash:
            return {"valid": False, "rows_checked": expected_seq, "message": "Audit order or predecessor mismatch"}
        recomputed = hashlib.sha256(canonical(_fields(event))).hexdigest()
        if recomputed != event.event_hash or not hmac.compare_digest(
            signature(recomputed.encode("ascii")), event.event_hmac
        ):
            return {"valid": False, "rows_checked": expected_seq, "message": "Audit hash or HMAC mismatch"}
        checkpoint = checkpoints.pop(event.sequence_number, None)
        if checkpoint is not None:
            check_payload = canonical({"sequence": event.sequence_number, "hash": recomputed})
            if checkpoint.last_event_hash != recomputed or not hmac.compare_digest(
                checkpoint.checkpoint_hmac, signature(check_payload)
            ):
                return {"valid": False, "rows_checked": expected_seq, "message": "Checkpoint mismatch"}
        expected_seq, expected_hash = event.sequence_number, recomputed
    head = session.get(AuditHead, 1)
    if checkpoints or head is None or head.last_sequence != expected_seq or head.last_hash != expected_hash:
        return {"valid": False, "rows_checked": expected_seq, "message": "Audit head or checkpoint mismatch"}
    return {"valid": True, "rows_checked": expected_seq, "message": "Audit chain verified"}
