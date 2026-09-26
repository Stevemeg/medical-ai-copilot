"""PostgreSQL schema and connection lifecycle; Alembic owns DDL."""

from datetime import date, datetime, timezone
from contextlib import contextmanager
from collections.abc import Iterator
from functools import lru_cache
from uuid import UUID, uuid4

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID as PGUUID
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker
from opentelemetry import trace

from backend.settings import get_settings


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Patient(Base):
    __tablename__ = "patients"
    patient_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    source_patient_id: Mapped[str] = mapped_column(String(200), unique=True, nullable=False)
    current_context_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    current_snapshot_id: Mapped[UUID | None] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("patient_snapshots.snapshot_id", ondelete="RESTRICT")
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class PatientSnapshot(Base):
    __tablename__ = "patient_snapshots"
    __table_args__ = (UniqueConstraint("patient_id", "context_hash"),)
    snapshot_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    patient_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("patients.patient_id", ondelete="RESTRICT"), nullable=False
    )
    context_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    normalized_context_json: Mapped[dict] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class ClinicalReviewRow(Base):
    __tablename__ = "clinical_reviews"
    review_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    patient_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("patients.patient_id", ondelete="RESTRICT"), nullable=False
    )
    snapshot_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("patient_snapshots.snapshot_id", ondelete="RESTRICT"), nullable=False
    )
    patient_context_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    review_json: Mapped[dict] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    snapshot: Mapped[PatientSnapshot] = relationship()


class ClinicalFindingRow(Base):
    __tablename__ = "clinical_findings"
    __table_args__ = (UniqueConstraint("review_id", "rule_id", "rule_version"),)
    finding_id: Mapped[str] = mapped_column(String(80), primary_key=True)
    review_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("clinical_reviews.review_id", ondelete="RESTRICT"), nullable=False
    )
    rule_id: Mapped[str] = mapped_column(String(120), nullable=False)
    rule_version: Mapped[str] = mapped_column(String(80), nullable=False)
    status: Mapped[str] = mapped_column(String(40), nullable=False)
    patient_context_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    evaluated_as_of: Mapped[date] = mapped_column(Date, nullable=False)
    finding_payload: Mapped[dict] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class FindingActionRow(Base):
    __tablename__ = "finding_actions"
    __table_args__ = (
        CheckConstraint(
            "action_type IN ('accept','dismiss','already_addressed','incorrect_evidence','not_clinically_relevant')"
        ),
    )
    action_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    finding_id: Mapped[str] = mapped_column(
        String(80), ForeignKey("clinical_findings.finding_id", ondelete="RESTRICT"), nullable=False
    )
    action_type: Mapped[str] = mapped_column(String(40), nullable=False)
    note: Mapped[str | None] = mapped_column(String(2000))
    actor_id: Mapped[str | None] = mapped_column(String(200))
    action_json: Mapped[dict] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class AuditHead(Base):
    __tablename__ = "audit_head"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    last_sequence: Mapped[int] = mapped_column(BigInteger, nullable=False)
    last_hash: Mapped[str] = mapped_column(String(64), nullable=False)


class AuditEvent(Base):
    __tablename__ = "audit_events"
    sequence_number: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    audit_event_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), unique=True, default=uuid4, nullable=False)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    actor_subject: Mapped[str] = mapped_column(String(200), nullable=False)
    actor_roles: Mapped[list] = mapped_column(JSONB, nullable=False)
    request_id: Mapped[str] = mapped_column(String(100), nullable=False)
    event_type: Mapped[str] = mapped_column(String(100), nullable=False)
    resource_type: Mapped[str] = mapped_column(String(100), nullable=False)
    resource_id: Mapped[str] = mapped_column(String(200), nullable=False)
    event_payload: Mapped[dict] = mapped_column(JSONB, nullable=False)
    previous_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    event_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    event_hmac: Mapped[str] = mapped_column(String(64), nullable=False)


class AuditCheckpoint(Base):
    __tablename__ = "audit_checkpoints"
    last_sequence: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    last_event_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    checkpoint_hmac: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class IdempotencyKey(Base):
    __tablename__ = "idempotency_keys"
    __table_args__ = (UniqueConstraint("actor_subject", "operation", "key"),)
    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    actor_subject: Mapped[str] = mapped_column(String(200), nullable=False)
    operation: Mapped[str] = mapped_column(String(100), nullable=False)
    key: Mapped[str] = mapped_column(String(200), nullable=False)
    request_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    response_json: Mapped[dict] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class RateBucket(Base):
    __tablename__ = "rate_buckets"
    __table_args__ = (UniqueConstraint("actor_subject", "operation", "minute"),)
    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    actor_subject: Mapped[str] = mapped_column(String(200), nullable=False)
    operation: Mapped[str] = mapped_column(String(100), nullable=False)
    minute: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    count: Mapped[int] = mapped_column(Integer, nullable=False)


@lru_cache
def engine():
    settings = get_settings()
    return create_engine(
        settings.database_url,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        pool_timeout=settings.db_pool_timeout,
        pool_pre_ping=True,
        pool_recycle=1800,
    )


@lru_cache
def session_factory():
    return sessionmaker(engine(), expire_on_commit=False)


@contextmanager
def transaction() -> Iterator[Session]:
    with trace.get_tracer(__name__).start_as_current_span("database.transaction"):
        with session_factory().begin() as session:
            yield session
