"""Transactional PostgreSQL repository for mutable clinical application state."""

import hashlib
import json
from uuid import UUID, uuid4

from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert

from backend.clinical_models import ActionType, ClinicalFinding, FindingAction
from backend.db import (
    ClinicalFindingRow,
    ClinicalReviewRow,
    FindingActionRow,
    IdempotencyKey,
    Patient,
    PatientSnapshot,
    session_factory,
    transaction,
    utcnow,
)
from backend.durable_audit import append_event
from backend.patient_context import context_hash, data_availability, timeline
from backend.patient_models import ClinicalReview, PatientContext, ReviewStatus
from backend.patient_store import FindingNotFound, PatientNotFound, ReviewCompleted, ReviewNotFound


def _uuid(value: str, not_found: type[LookupError] | None = None) -> UUID:
    try:
        return UUID(value)
    except ValueError as exc:
        if not_found is not None:
            raise not_found(value) from exc
        raise


class IdempotencyConflict(ValueError):
    pass


class PostgresPatientRepository:
    def __init__(
        self, *, actor_subject: str = "system", actor_roles: list[str] | None = None, request_id: str = "system"
    ):
        self.actor_subject = actor_subject
        self.actor_roles = actor_roles or []
        self.request_id = request_id

    def _audit(self, session, event_type: str, resource_type: str, resource_id: str, payload: dict | None = None):
        append_event(
            session,
            event_type=event_type,
            resource_type=resource_type,
            resource_id=resource_id,
            payload=payload,
            actor_subject=self.actor_subject,
            actor_roles=self.actor_roles,
            request_id=self.request_id,
        )

    def _idempotent_get(self, session, operation: str, key: str | None, fingerprint: str) -> dict | None:
        if key is None:
            return None
        if not 1 <= len(key) <= 200:
            raise IdempotencyConflict("Invalid idempotency key")
        lock = int.from_bytes(
            hashlib.sha256(f"{self.actor_subject}:{operation}:{key}".encode()).digest()[:8], "big", signed=True
        )
        session.execute(text("SELECT pg_advisory_xact_lock(:lock)"), {"lock": lock})
        row = session.scalar(
            select(IdempotencyKey).where(
                IdempotencyKey.actor_subject == self.actor_subject,
                IdempotencyKey.operation == operation,
                IdempotencyKey.key == key,
            )
        )
        if row is None:
            return None
        if row.request_fingerprint != fingerprint:
            raise IdempotencyConflict("Idempotency key reused with a different request")
        return row.response_json

    def _idempotent_save(self, session, operation: str, key: str | None, fingerprint: str, response: dict) -> None:
        if key:
            session.add(
                IdempotencyKey(
                    actor_subject=self.actor_subject,
                    operation=operation,
                    key=key,
                    request_fingerprint=fingerprint,
                    response_json=response,
                )
            )

    @staticmethod
    def _fingerprint(value: dict) -> str:
        return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def import_context(self, context: PatientContext, idempotency_key: str | None = None) -> tuple[str, str, bool]:
        digest = context_hash(context)
        source_id = context.patient.source_patient_id
        with transaction() as session:
            fingerprint = self._fingerprint({"source_id": source_id, "context_hash": digest})
            replay = self._idempotent_get(session, "patient_import", idempotency_key, fingerprint)
            if replay is not None:
                return replay["patient_id"], replay["context_hash"], replay["changed"]
            # ON CONFLICT first establishes one stable identity; row lock serializes context updates.
            stmt = (
                insert(Patient)
                .values(patient_id=uuid4(), source_patient_id=source_id, current_context_hash=digest)
                .on_conflict_do_nothing(index_elements=[Patient.source_patient_id])
                .returning(Patient.patient_id)
            )
            inserted = session.scalar(stmt) is not None
            patient = session.scalar(select(Patient).where(Patient.source_patient_id == source_id).with_for_update())
            if patient is None:
                raise ValueError("Patient upsert failed")
            snapshot = session.scalar(
                select(PatientSnapshot).where(
                    PatientSnapshot.patient_id == patient.patient_id, PatientSnapshot.context_hash == digest
                )
            )
            changed = patient.current_snapshot_id is None or snapshot is None or patient.current_context_hash != digest
            if snapshot is None:
                snapshot = PatientSnapshot(
                    patient_id=patient.patient_id,
                    context_hash=digest,
                    normalized_context_json=context.model_dump(mode="json"),
                )
                session.add(snapshot)
                session.flush()
            if changed:
                patient.current_snapshot_id = snapshot.snapshot_id
                patient.current_context_hash = digest
                patient.updated_at = utcnow()
                self._audit(
                    session,
                    "patient_imported" if inserted else "patient_updated",
                    "patient",
                    str(patient.patient_id),
                    {"context_hash": digest},
                )
            self._idempotent_save(
                session,
                "patient_import",
                idempotency_key,
                fingerprint,
                {"patient_id": str(patient.patient_id), "context_hash": digest, "changed": changed},
            )
            return str(patient.patient_id), digest, changed

    def get_patient(self, patient_id: str) -> tuple[PatientContext, str]:
        with session_factory()() as session:
            patient = session.get(Patient, _uuid(patient_id, PatientNotFound))
            if patient is None or patient.current_snapshot_id is None:
                raise PatientNotFound(patient_id)
            snapshot = session.get(PatientSnapshot, patient.current_snapshot_id)
            if snapshot is None:
                raise ValueError("Current patient snapshot missing")
            context = PatientContext.model_validate(snapshot.normalized_context_json)
            if context_hash(context) != snapshot.context_hash or snapshot.context_hash != patient.current_context_hash:
                raise ValueError("Patient snapshot integrity mismatch")
            return context, snapshot.context_hash

    def list_patients(self, limit: int = 50, cursor: str | None = None) -> list[tuple[str, PatientContext, str]]:
        with session_factory()() as session:
            stmt = select(Patient).order_by(Patient.source_patient_id).limit(limit)
            if cursor:
                stmt = stmt.where(Patient.source_patient_id > cursor)
            patients = session.scalars(stmt).all()
            rows = []
            for patient in patients:
                snapshot = session.get(PatientSnapshot, patient.current_snapshot_id)
                if snapshot is None:
                    raise ValueError("Current patient snapshot missing")
                context = PatientContext.model_validate(snapshot.normalized_context_json)
                rows.append((str(patient.patient_id), context, snapshot.context_hash))
            return rows

    def create_review(self, patient_id: str, idempotency_key: str | None = None) -> ClinicalReview:
        with transaction() as session:
            fingerprint = self._fingerprint({"patient_id": patient_id})
            replay = self._idempotent_get(session, "review_create", idempotency_key, fingerprint)
            if replay is not None:
                return ClinicalReview.model_validate(replay)
            patient = session.scalar(
                select(Patient).where(Patient.patient_id == _uuid(patient_id, PatientNotFound)).with_for_update()
            )
            if patient is None or patient.current_snapshot_id is None:
                raise PatientNotFound(patient_id)
            snapshot = session.get(PatientSnapshot, patient.current_snapshot_id)
            if snapshot is None:
                raise ValueError("Current patient snapshot missing")
            context = PatientContext.model_validate(snapshot.normalized_context_json)
            review = ClinicalReview(
                review_id=str(uuid4()),
                patient_id=patient_id,
                patient_context_hash=snapshot.context_hash,
                created_at=utcnow().isoformat(),
                status=ReviewStatus.READY_FOR_REVIEW,
                patient_snapshot=context,
                timeline=timeline(context),
                data_availability=data_availability(context),
            )
            session.add(
                ClinicalReviewRow(
                    review_id=_uuid(review.review_id),
                    patient_id=patient.patient_id,
                    snapshot_id=snapshot.snapshot_id,
                    patient_context_hash=snapshot.context_hash,
                    status=review.status.value,
                    review_json=review.model_dump(mode="json"),
                )
            )
            self._audit(
                session,
                "review_created",
                "review",
                review.review_id,
                {"patient_id": patient_id, "context_hash": snapshot.context_hash},
            )
            self._idempotent_save(
                session, "review_create", idempotency_key, fingerprint, review.model_dump(mode="json")
            )
            return review

    def get_review(self, review_id: str) -> ClinicalReview:
        with session_factory()() as session:
            row = session.get(ClinicalReviewRow, _uuid(review_id, ReviewNotFound))
            if row is None:
                raise ReviewNotFound(review_id)
            review = ClinicalReview.model_validate(row.review_json)
            if (
                review.patient_context_hash != row.patient_context_hash
                or context_hash(review.patient_snapshot) != row.snapshot.context_hash
            ):
                raise ValueError("Review snapshot integrity mismatch")
            return review

    def list_reviews(self, patient_id: str, limit: int = 50) -> list[ClinicalReview]:
        self.get_patient(patient_id)
        with session_factory()() as session:
            rows = session.scalars(
                select(ClinicalReviewRow)
                .where(ClinicalReviewRow.patient_id == _uuid(patient_id, PatientNotFound))
                .order_by(ClinicalReviewRow.created_at.desc())
                .limit(limit)
            ).all()
            return [ClinicalReview.model_validate(row.review_json) for row in rows]

    def complete_review(self, review_id: str) -> ClinicalReview:
        with transaction() as session:
            row = session.scalar(
                select(ClinicalReviewRow)
                .where(ClinicalReviewRow.review_id == _uuid(review_id, ReviewNotFound))
                .with_for_update()
            )
            if row is None:
                raise ReviewNotFound(review_id)
            review = ClinicalReview.model_validate(row.review_json)
            if review.status == ReviewStatus.COMPLETED:
                return review
            now = utcnow()
            updated = review.model_copy(update={"status": ReviewStatus.COMPLETED, "completed_at": now.isoformat()})
            row.status, row.completed_at, row.review_json = updated.status.value, now, updated.model_dump(mode="json")
            self._audit(session, "review_completed", "review", review_id)
            return updated

    def save_findings(
        self, review_id: str, findings: tuple[ClinicalFinding, ...], idempotency_key: str | None = None
    ) -> tuple[ClinicalFinding, ...]:
        with transaction() as session:
            fingerprint = self._fingerprint(
                {
                    "review_id": review_id,
                    "as_of": findings[0].evaluated_as_of.isoformat(),
                    "coverage": findings[0].record_coverage.model_dump(mode="json")
                    if findings[0].record_coverage
                    else None,
                }
            )
            replay = self._idempotent_get(session, "review_evaluate", idempotency_key, fingerprint)
            if replay is not None:
                return tuple(ClinicalFinding.model_validate(row) for row in replay["findings"])
            row = session.scalar(
                select(ClinicalReviewRow)
                .where(ClinicalReviewRow.review_id == _uuid(review_id, ReviewNotFound))
                .with_for_update()
            )
            if row is None:
                raise ReviewNotFound(review_id)
            if row.status == ReviewStatus.COMPLETED.value:
                raise ReviewCompleted(review_id)
            if any(f.review_id != review_id or f.patient_context_hash != row.patient_context_hash for f in findings):
                raise ValueError("Finding does not match review snapshot")
            existing = session.scalars(
                select(ClinicalFindingRow).where(ClinicalFindingRow.review_id == row.review_id)
            ).all()
            if existing:
                current = tuple(ClinicalFinding.model_validate(f.finding_payload) for f in existing)
                if not all(
                    f.evaluated_as_of == findings[0].evaluated_as_of
                    and f.record_coverage == findings[0].record_coverage
                    for f in current
                ):
                    raise ValueError("Review already evaluated with different context")
                result = self._sort(current)
                self._idempotent_save(
                    session,
                    "review_evaluate",
                    idempotency_key,
                    fingerprint,
                    {"findings": [f.model_dump(mode="json") for f in result]},
                )
                return result
            for finding in findings:
                session.add(
                    ClinicalFindingRow(
                        finding_id=finding.finding_id,
                        review_id=row.review_id,
                        rule_id=finding.rule_id,
                        rule_version=finding.rule_version,
                        status=finding.status.value,
                        patient_context_hash=finding.patient_context_hash,
                        evaluated_as_of=finding.evaluated_as_of,
                        finding_payload=finding.model_dump(mode="json"),
                        created_at=finding.created_at,
                    )
                )
            self._audit(
                session,
                "review_evaluated",
                "review",
                review_id,
                {"finding_count": len(findings), "statuses": [f.status.value for f in findings]},
            )
            result = self._sort(findings)
            self._idempotent_save(
                session,
                "review_evaluate",
                idempotency_key,
                fingerprint,
                {"findings": [f.model_dump(mode="json") for f in result]},
            )
            return result

    @staticmethod
    def _sort(findings: tuple[ClinicalFinding, ...]) -> tuple[ClinicalFinding, ...]:
        priority = {
            "potential_care_gap": 0,
            "insufficient_data": 1,
            "satisfied": 2,
            "suppressed": 3,
            "not_applicable": 4,
        }
        return tuple(sorted(findings, key=lambda f: (priority[f.status.value], f.rule_id)))

    def list_findings(self, review_id: str, limit: int = 100) -> tuple[ClinicalFinding, ...]:
        review = self.get_review(review_id)
        with session_factory()() as session:
            rows = session.scalars(
                select(ClinicalFindingRow)
                .where(ClinicalFindingRow.review_id == _uuid(review_id, ReviewNotFound))
                .limit(limit)
            ).all()
            findings = tuple(ClinicalFinding.model_validate(row.finding_payload) for row in rows)
            if any(
                f.patient_context_hash != row.patient_context_hash
                or f.patient_context_hash != review.patient_context_hash
                for f, row in zip(findings, rows)
            ):
                raise ValueError("Stored finding/review snapshot mismatch")
            return self._sort(findings)

    def get_finding(self, finding_id: str) -> ClinicalFinding:
        with session_factory()() as session:
            row = session.get(ClinicalFindingRow, finding_id)
            if row is None:
                raise FindingNotFound(finding_id)
            finding = ClinicalFinding.model_validate(row.finding_payload)
            review = session.get(ClinicalReviewRow, row.review_id)
            if (
                review is None
                or finding.patient_context_hash != row.patient_context_hash
                or row.patient_context_hash != review.patient_context_hash
            ):
                raise ValueError("Stored finding/review snapshot mismatch")
            return finding

    def add_action(
        self, finding_id: str, action_type: ActionType, note: str | None = None, idempotency_key: str | None = None
    ) -> FindingAction:
        with transaction() as session:
            fingerprint = self._fingerprint({"finding_id": finding_id, "action_type": action_type.value, "note": note})
            replay = self._idempotent_get(session, "finding_action", idempotency_key, fingerprint)
            if replay is not None:
                return FindingAction.model_validate(replay)
            finding = session.get(ClinicalFindingRow, finding_id)
            if finding is None:
                raise FindingNotFound(finding_id)
            action = FindingAction(
                action_id=str(uuid4()),
                finding_id=finding_id,
                action_type=action_type,
                note=note,
                created_at=utcnow(),
                actor=self.actor_subject,
            )
            session.add(
                FindingActionRow(
                    action_id=_uuid(action.action_id),
                    finding_id=finding_id,
                    action_type=action_type.value,
                    note=note,
                    actor_id=self.actor_subject,
                    action_json=action.model_dump(mode="json"),
                )
            )
            self._audit(
                session,
                "finding_action_recorded",
                "finding",
                finding_id,
                {"action_id": action.action_id, "action_type": action_type.value},
            )
            self._idempotent_save(
                session, "finding_action", idempotency_key, fingerprint, action.model_dump(mode="json")
            )
            return action

    def list_actions(self, finding_id: str, limit: int = 100) -> tuple[FindingAction, ...]:
        self.get_finding(finding_id)
        with session_factory()() as session:
            rows = session.scalars(
                select(FindingActionRow)
                .where(FindingActionRow.finding_id == finding_id)
                .order_by(FindingActionRow.created_at, FindingActionRow.action_id)
                .limit(limit)
            ).all()
            return tuple(FindingAction.model_validate(row.action_json) for row in rows)
