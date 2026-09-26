"""Initial PostgreSQL operational schema

Revision ID: 411c286b9ec8
Revises:
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "411c286b9ec8"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "audit_checkpoints",
        sa.Column("last_sequence", sa.BigInteger(), nullable=False),
        sa.Column("last_event_hash", sa.String(length=64), nullable=False),
        sa.Column("checkpoint_hmac", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("last_sequence"),
    )
    op.create_table(
        "audit_events",
        sa.Column("sequence_number", sa.BigInteger(), nullable=False),
        sa.Column("audit_event_id", sa.UUID(), nullable=False),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("actor_subject", sa.String(length=200), nullable=False),
        sa.Column("actor_roles", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("request_id", sa.String(length=100), nullable=False),
        sa.Column("event_type", sa.String(length=100), nullable=False),
        sa.Column("resource_type", sa.String(length=100), nullable=False),
        sa.Column("resource_id", sa.String(length=200), nullable=False),
        sa.Column("event_payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("previous_hash", sa.String(length=64), nullable=False),
        sa.Column("event_hash", sa.String(length=64), nullable=False),
        sa.Column("event_hmac", sa.String(length=64), nullable=False),
        sa.PrimaryKeyConstraint("sequence_number"),
        sa.UniqueConstraint("audit_event_id"),
    )
    op.create_table(
        "audit_head",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("last_sequence", sa.BigInteger(), nullable=False),
        sa.Column("last_hash", sa.String(length=64), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "idempotency_keys",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("actor_subject", sa.String(length=200), nullable=False),
        sa.Column("operation", sa.String(length=100), nullable=False),
        sa.Column("key", sa.String(length=200), nullable=False),
        sa.Column("request_fingerprint", sa.String(length=64), nullable=False),
        sa.Column("response_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("actor_subject", "operation", "key"),
    )
    op.create_table(
        "patients",
        sa.Column("patient_id", sa.UUID(), nullable=False),
        sa.Column("source_patient_id", sa.String(length=200), nullable=False),
        sa.Column("current_context_hash", sa.String(length=64), nullable=False),
        sa.Column("current_snapshot_id", sa.UUID(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("patient_id"),
        sa.UniqueConstraint("source_patient_id"),
    )
    op.create_table(
        "patient_snapshots",
        sa.Column("snapshot_id", sa.UUID(), nullable=False),
        sa.Column("patient_id", sa.UUID(), nullable=False),
        sa.Column("context_hash", sa.String(length=64), nullable=False),
        sa.Column("normalized_context_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["patient_id"], ["patients.patient_id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("snapshot_id"),
        sa.UniqueConstraint("patient_id", "context_hash"),
    )
    op.create_foreign_key(
        "fk_patients_current_snapshot",
        "patients",
        "patient_snapshots",
        ["current_snapshot_id"],
        ["snapshot_id"],
        ondelete="RESTRICT",
    )
    op.create_table(
        "rate_buckets",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("actor_subject", sa.String(length=200), nullable=False),
        sa.Column("operation", sa.String(length=100), nullable=False),
        sa.Column("minute", sa.DateTime(timezone=True), nullable=False),
        sa.Column("count", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("actor_subject", "operation", "minute"),
    )
    op.create_table(
        "clinical_reviews",
        sa.Column("review_id", sa.UUID(), nullable=False),
        sa.Column("patient_id", sa.UUID(), nullable=False),
        sa.Column("snapshot_id", sa.UUID(), nullable=False),
        sa.Column("patient_context_hash", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("review_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["patient_id"], ["patients.patient_id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["snapshot_id"], ["patient_snapshots.snapshot_id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("review_id"),
    )
    op.create_table(
        "clinical_findings",
        sa.Column("finding_id", sa.String(length=80), nullable=False),
        sa.Column("review_id", sa.UUID(), nullable=False),
        sa.Column("rule_id", sa.String(length=120), nullable=False),
        sa.Column("rule_version", sa.String(length=80), nullable=False),
        sa.Column("status", sa.String(length=40), nullable=False),
        sa.Column("patient_context_hash", sa.String(length=64), nullable=False),
        sa.Column("evaluated_as_of", sa.Date(), nullable=False),
        sa.Column("finding_payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["review_id"], ["clinical_reviews.review_id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("finding_id"),
        sa.UniqueConstraint("review_id", "rule_id", "rule_version"),
    )
    op.create_table(
        "finding_actions",
        sa.Column("action_id", sa.UUID(), nullable=False),
        sa.Column("finding_id", sa.String(length=80), nullable=False),
        sa.Column("action_type", sa.String(length=40), nullable=False),
        sa.Column("note", sa.String(length=2000), nullable=True),
        sa.Column("actor_id", sa.String(length=200), nullable=True),
        sa.Column("action_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "action_type IN ('accept','dismiss','already_addressed','incorrect_evidence','not_clinically_relevant')"
        ),
        sa.ForeignKeyConstraint(["finding_id"], ["clinical_findings.finding_id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("action_id"),
    )
    op.execute("INSERT INTO audit_head (id, last_sequence, last_hash) VALUES (1, 0, '" + "0" * 64 + "')")
    op.execute("""
        CREATE FUNCTION reject_immutable_mutation() RETURNS trigger AS $$
        BEGIN RAISE EXCEPTION 'Historical rows are immutable'; END;
        $$ LANGUAGE plpgsql
    """)
    for table in ("audit_events", "audit_checkpoints", "patient_snapshots", "clinical_findings", "finding_actions"):
        op.execute(
            f"CREATE TRIGGER {table}_append_only BEFORE UPDATE OR DELETE ON {table} "
            "FOR EACH ROW EXECUTE FUNCTION reject_immutable_mutation()"
        )
        op.execute(
            f"CREATE TRIGGER {table}_no_truncate BEFORE TRUNCATE ON {table} "
            "FOR EACH STATEMENT EXECUTE FUNCTION reject_immutable_mutation()"
        )


def downgrade() -> None:
    # Destruction of clinical/audit history is intentionally disallowed.
    connection = op.get_bind()
    for table in (
        "patients",
        "patient_snapshots",
        "clinical_reviews",
        "clinical_findings",
        "finding_actions",
        "audit_events",
        "audit_checkpoints",
        "idempotency_keys",
    ):
        if connection.scalar(sa.text(f"SELECT EXISTS (SELECT 1 FROM {table} LIMIT 1)")):
            raise RuntimeError("Downgrade is allowed only for an unused empty schema")
    for table in ("audit_events", "audit_checkpoints", "patient_snapshots", "clinical_findings", "finding_actions"):
        op.execute(f"DROP TRIGGER {table}_append_only ON {table}")
        op.execute(f"DROP TRIGGER {table}_no_truncate ON {table}")
    op.execute("DROP FUNCTION reject_immutable_mutation()")
    op.drop_table("finding_actions")
    op.drop_table("clinical_findings")
    op.drop_table("clinical_reviews")
    op.drop_table("rate_buckets")
    op.drop_constraint("fk_patients_current_snapshot", "patients", type_="foreignkey")
    op.drop_table("patient_snapshots")
    op.drop_table("patients")
    op.drop_table("idempotency_keys")
    op.drop_table("audit_head")
    op.drop_table("audit_events")
    op.drop_table("audit_checkpoints")
