"""SQLite persistence for normalized synthetic patient snapshots and reviews."""

from __future__ import annotations

import sqlite3
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from backend.patient_context import context_hash, data_availability, timeline
from backend.patient_models import ClinicalReview, PatientContext, ReviewStatus
from backend.clinical_models import ClinicalFinding, FindingAction, ActionType

LOG = logging.getLogger(__name__)


class FindingNotFound(LookupError):
    pass


class ReviewCompleted(ValueError):
    pass


class PatientNotFound(LookupError):
    pass


class ReviewNotFound(LookupError):
    pass


class PatientRepository(Protocol):
    def import_context(self, context: PatientContext) -> tuple[str, str, bool]: ...
    def get_patient(self, patient_id: str) -> tuple[PatientContext, str]: ...
    def list_patients(self) -> list[tuple[str, PatientContext, str]]: ...
    def create_review(self, patient_id: str) -> ClinicalReview: ...
    def get_review(self, review_id: str) -> ClinicalReview: ...
    def list_reviews(self, patient_id: str) -> list[ClinicalReview]: ...
    def complete_review(self, review_id: str) -> ClinicalReview: ...


class SQLitePatientRepository:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS patients (
                    patient_id TEXT PRIMARY KEY,
                    source_patient_id TEXT NOT NULL UNIQUE,
                    context_hash TEXT NOT NULL,
                    context_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS reviews (
                    review_id TEXT PRIMARY KEY,
                    patient_id TEXT NOT NULL,
                    context_hash TEXT NOT NULL,
                    review_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(patient_id) REFERENCES patients(patient_id)
                );
                CREATE TABLE IF NOT EXISTS clinical_findings (
                    finding_id TEXT PRIMARY KEY,
                    review_id TEXT NOT NULL REFERENCES reviews(review_id),
                    rule_id TEXT NOT NULL,
                    rule_version TEXT NOT NULL,
                    context_hash TEXT NOT NULL,
                    finding_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(review_id, rule_id, rule_version)
                );
                CREATE TABLE IF NOT EXISTS finding_actions (
                    action_id TEXT PRIMARY KEY,
                    finding_id TEXT NOT NULL REFERENCES clinical_findings(finding_id),
                    action_type TEXT NOT NULL CHECK(action_type IN ('accept','dismiss','already_addressed','incorrect_evidence','not_clinically_relevant')),
                    action_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
            """)

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=10)
        db.execute("PRAGMA foreign_keys=ON")
        return db

    def import_context(self, context: PatientContext) -> tuple[str, str, bool]:
        digest = context_hash(context)
        with self._connect() as db:
            existing = db.execute(
                "SELECT patient_id, context_hash FROM patients WHERE source_patient_id=?",
                (context.patient.source_patient_id,),
            ).fetchone()
            patient_id = existing[0] if existing else str(uuid4())
            changed = existing is None or existing[1] != digest
            if changed:
                db.execute(
                    """INSERT INTO patients (patient_id, source_patient_id, context_hash, context_json, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(source_patient_id) DO UPDATE SET context_hash=excluded.context_hash,
                    context_json=excluded.context_json, updated_at=excluded.updated_at""",
                    (
                        patient_id,
                        context.patient.source_patient_id,
                        digest,
                        context.model_dump_json(),
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )
        return patient_id, digest, changed

    def get_patient(self, patient_id: str) -> tuple[PatientContext, str]:
        with self._connect() as db:
            row = db.execute(
                "SELECT context_json, context_hash FROM patients WHERE patient_id=?", (patient_id,)
            ).fetchone()
        if row is None:
            raise PatientNotFound(patient_id)
        context = PatientContext.model_validate_json(row[0])
        if context_hash(context) != row[1]:
            raise ValueError("Stored patient context hash mismatch")
        return context, row[1]

    def list_patients(self, limit: int = 50, cursor: str | None = None) -> list[tuple[str, PatientContext, str]]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT patient_id, context_json, context_hash FROM patients ORDER BY source_patient_id"
            ).fetchall()
        patients = [(row[0], PatientContext.model_validate_json(row[1]), row[2]) for row in rows]
        if any(context_hash(context) != digest for _, context, digest in patients):
            raise ValueError("Stored patient context hash mismatch")
        return [row for row in patients if cursor is None or row[1].patient.source_patient_id > cursor][:limit]

    def create_review(self, patient_id: str) -> ClinicalReview:
        context, digest = self.get_patient(patient_id)
        now = datetime.now(timezone.utc).isoformat()
        review = ClinicalReview(
            review_id=str(uuid4()),
            patient_id=patient_id,
            patient_context_hash=digest,
            created_at=now,
            status=ReviewStatus.READY_FOR_REVIEW,
            patient_snapshot=context,
            timeline=timeline(context),
            data_availability=data_availability(context),
        )
        with self._connect() as db:
            db.execute(
                "INSERT INTO reviews VALUES (?, ?, ?, ?, ?)",
                (review.review_id, patient_id, digest, review.model_dump_json(), now),
            )
        return review

    def get_review(self, review_id: str) -> ClinicalReview:
        with self._connect() as db:
            row = db.execute("SELECT review_json, context_hash FROM reviews WHERE review_id=?", (review_id,)).fetchone()
        if row is None:
            raise ReviewNotFound(review_id)
        review = ClinicalReview.model_validate_json(row[0])
        if review.patient_context_hash != row[1] or context_hash(review.patient_snapshot) != row[1]:
            raise ValueError("Stored review snapshot hash mismatch")
        return review

    def list_reviews(self, patient_id: str, limit: int = 50) -> list[ClinicalReview]:
        self.get_patient(patient_id)
        with self._connect() as db:
            rows = db.execute(
                "SELECT review_id FROM reviews WHERE patient_id=? ORDER BY created_at DESC, review_id",
                (patient_id,),
            ).fetchall()
        return [self.get_review(row[0]) for row in rows[:limit]]

    def complete_review(self, review_id: str) -> ClinicalReview:
        review = self.get_review(review_id)
        if review.status == ReviewStatus.COMPLETED:
            return review
        updated = review.model_copy(
            update={
                "status": ReviewStatus.COMPLETED,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        with self._connect() as db:
            db.execute("UPDATE reviews SET review_json=? WHERE review_id=?", (updated.model_dump_json(), review_id))
        return updated

    def save_findings(self, review_id: str, findings: tuple[ClinicalFinding, ...]) -> tuple[ClinicalFinding, ...]:
        review = self.get_review(review_id)
        if review.status == ReviewStatus.COMPLETED:
            raise ReviewCompleted(review_id)
        if any(f.review_id != review_id or f.patient_context_hash != review.patient_context_hash for f in findings):
            raise ValueError("Finding does not match review snapshot")
        with self._connect() as db:
            for finding in findings:
                existing = db.execute(
                    "SELECT finding_json FROM clinical_findings WHERE review_id=? AND rule_id=? AND rule_version=?",
                    (review_id, finding.rule_id, finding.rule_version),
                ).fetchone()
                if existing is not None:
                    if ClinicalFinding.model_validate_json(existing[0]).model_dump(
                        exclude={"created_at"}
                    ) != finding.model_dump(exclude={"created_at"}):
                        raise ValueError("Review already evaluated with different context; create a new review")
                    continue
                db.execute(
                    "INSERT INTO clinical_findings VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        finding.finding_id,
                        review_id,
                        finding.rule_id,
                        finding.rule_version,
                        finding.patient_context_hash,
                        finding.model_dump_json(),
                        finding.created_at.isoformat(),
                    ),
                )
                LOG.info(
                    "finding_created review_id=%s finding_id=%s rule_id=%s",
                    review_id,
                    finding.finding_id,
                    finding.rule_id,
                )
        return self.list_findings(review_id)

    def list_findings(self, review_id: str, limit: int = 100) -> tuple[ClinicalFinding, ...]:
        review = self.get_review(review_id)
        with self._connect() as db:
            rows = db.execute(
                "SELECT finding_json, context_hash FROM clinical_findings WHERE review_id=?", (review_id,)
            ).fetchall()
        findings = tuple(ClinicalFinding.model_validate_json(row[0]) for row in rows)
        if any(
            f.review_id != review.review_id
            or f.patient_context_hash != review.patient_context_hash
            or f.patient_context_hash != row[1]
            for f, row in zip(findings, rows)
        ):
            raise ValueError("Stored finding/review snapshot mismatch")
        priority = {
            "potential_care_gap": 0,
            "insufficient_data": 1,
            "satisfied": 2,
            "suppressed": 3,
            "not_applicable": 4,
        }
        return tuple(sorted(findings, key=lambda f: (priority[f.status.value], f.rule_id)))[:limit]

    def get_finding(self, finding_id: str) -> ClinicalFinding:
        with self._connect() as db:
            row = db.execute(
                "SELECT review_id, finding_json, context_hash FROM clinical_findings WHERE finding_id=?", (finding_id,)
            ).fetchone()
        if row is None:
            raise FindingNotFound(finding_id)
        finding = ClinicalFinding.model_validate_json(row[1])
        review = self.get_review(row[0])
        if (
            finding.review_id != review.review_id
            or finding.patient_context_hash != review.patient_context_hash
            or finding.patient_context_hash != row[2]
        ):
            raise ValueError("Stored finding/review snapshot mismatch")
        return finding

    def add_action(self, finding_id: str, action_type: ActionType, note: str | None = None) -> FindingAction:
        self.get_finding(finding_id)
        action = FindingAction(
            action_id=str(uuid4()),
            finding_id=finding_id,
            action_type=action_type,
            note=note,
            created_at=datetime.now(timezone.utc),
        )
        with self._connect() as db:
            db.execute(
                "INSERT INTO finding_actions VALUES (?, ?, ?, ?, ?)",
                (
                    action.action_id,
                    finding_id,
                    action.action_type.value,
                    action.model_dump_json(),
                    action.created_at.isoformat(),
                ),
            )
        LOG.info(
            "finding_action_recorded finding_id=%s action_id=%s action_type=%s",
            finding_id,
            action.action_id,
            action_type.value,
        )
        return action

    def list_actions(self, finding_id: str, limit: int = 100) -> tuple[FindingAction, ...]:
        self.get_finding(finding_id)
        with self._connect() as db:
            rows = db.execute(
                "SELECT action_json FROM finding_actions WHERE finding_id=? ORDER BY created_at, action_id",
                (finding_id,),
            ).fetchall()
        return tuple(FindingAction.model_validate_json(row[0]) for row in rows[:limit])
