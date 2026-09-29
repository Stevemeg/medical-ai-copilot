"""Guideline quarantine, immutable approved payloads and atomic active pointer.

Revision ID: 62a2guidelines
Revises: 62a1phase6
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "62a2guidelines"
down_revision = "62a1phase6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "guideline_candidates",
        sa.Column("candidate_id", sa.UUID(), primary_key=True),
        sa.Column("document_id", sa.String(200), nullable=False),
        sa.Column("version_id", sa.String(200), unique=True, nullable=False),
        sa.Column("base_version_id", sa.String(200)),
        sa.Column("status", sa.String(40), nullable=False),
        sa.Column("checksum", sa.String(64), nullable=False),
        sa.Column("payload", JSONB(), nullable=False),
        sa.Column("diff", JSONB(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "guideline_activations",
        sa.Column("document_id", sa.String(200), primary_key=True),
        sa.Column("candidate_id", sa.UUID(), sa.ForeignKey("guideline_candidates.candidate_id"), nullable=False),
        sa.Column("activated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.execute("""CREATE FUNCTION protect_guideline_candidate() RETURNS trigger AS $$
        BEGIN
          IF TG_OP = 'UPDATE' AND NEW.status IS DISTINCT FROM OLD.status AND NOT (
            (OLD.status = 'quarantined' AND NEW.status = 'review_required') OR
            (OLD.status = 'review_required' AND NEW.status IN ('approved','rejected')) OR
            (OLD.status = 'approved' AND NEW.status = 'activated'))
          THEN RAISE EXCEPTION 'Invalid guideline workflow transition'; END IF;
          IF TG_OP = 'DELETE' OR (OLD.status IN ('approved','activated','rejected') AND
              (NEW.payload IS DISTINCT FROM OLD.payload OR NEW.checksum IS DISTINCT FROM OLD.checksum OR
               NEW.diff IS DISTINCT FROM OLD.diff OR NEW.version_id IS DISTINCT FROM OLD.version_id OR
               NEW.document_id IS DISTINCT FROM OLD.document_id OR NEW.base_version_id IS DISTINCT FROM OLD.base_version_id))
          THEN RAISE EXCEPTION 'Governed guideline history is immutable'; END IF;
          RETURN NEW;
        END; $$ LANGUAGE plpgsql""")
    op.execute(
        "CREATE TRIGGER guideline_candidate_protection BEFORE UPDATE OR DELETE ON guideline_candidates FOR EACH ROW EXECUTE FUNCTION protect_guideline_candidate()"
    )
    op.execute(
        "CREATE TRIGGER guideline_history_no_truncate BEFORE TRUNCATE ON guideline_candidates FOR EACH STATEMENT EXECUTE FUNCTION reject_immutable_mutation()"
    )


def downgrade() -> None:
    if op.get_bind().scalar(sa.text("SELECT EXISTS (SELECT 1 FROM guideline_candidates)")):
        raise RuntimeError("Guideline history cannot be discarded by downgrade")
    op.drop_table("guideline_activations")
    op.drop_table("guideline_candidates")
    op.execute("DROP FUNCTION protect_guideline_candidate()")
