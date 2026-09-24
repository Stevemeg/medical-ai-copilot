"""SQLite persistence for normalized synthetic patient snapshots and reviews."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from backend.patient_context import context_hash, data_availability, timeline
from backend.patient_models import ClinicalReview, PatientContext, ReviewStatus


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

    def list_patients(self) -> list[tuple[str, PatientContext, str]]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT patient_id, context_json, context_hash FROM patients ORDER BY source_patient_id"
            ).fetchall()
        patients = [(row[0], PatientContext.model_validate_json(row[1]), row[2]) for row in rows]
        if any(context_hash(context) != digest for _, context, digest in patients):
            raise ValueError("Stored patient context hash mismatch")
        return patients

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

    def list_reviews(self, patient_id: str) -> list[ClinicalReview]:
        self.get_patient(patient_id)
        with self._connect() as db:
            rows = db.execute(
                "SELECT review_id FROM reviews WHERE patient_id=? ORDER BY created_at DESC, review_id",
                (patient_id,),
            ).fetchall()
        return [self.get_review(row[0]) for row in rows]

    def complete_review(self, review_id: str) -> ClinicalReview:
        review = self.get_review(review_id)
        if review.status == ReviewStatus.COMPLETED:
            return review
        updated = review.model_copy(
            update={
                "status": ReviewStatus.COMPLETED,
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "clinician_actions": review.clinician_actions + ("marked_completed",),
            }
        )
        with self._connect() as db:
            db.execute("UPDATE reviews SET review_json=? WHERE review_id=?", (updated.model_dump_json(), review_id))
        return updated
