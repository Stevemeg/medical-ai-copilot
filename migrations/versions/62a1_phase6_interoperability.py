"""Short-lived SMART launch state and indexed operational retention.

Revision ID: 62a1phase6
Revises: 411c286b9ec8
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "62a1phase6"
down_revision = "411c286b9ec8"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "smart_launches",
        sa.Column("state_hash", sa.String(64), primary_key=True),
        sa.Column("browser_hash", sa.String(64), nullable=False),
        sa.Column("issuer", sa.String(1000), nullable=False),
        sa.Column("token_endpoint", sa.String(1000), nullable=False),
        sa.Column("verifier", sa.String(128)),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("consumed", sa.Boolean(), nullable=False),
        sa.Column("context_json", JSONB()),
    )
    op.create_index("ix_rate_buckets_minute", "rate_buckets", ["minute"])
    op.create_index("ix_idempotency_created", "idempotency_keys", ["created_at"])
    op.create_index("ix_smart_expiry", "smart_launches", ["expires_at"])


def downgrade() -> None:
    if op.get_bind().scalar(sa.text("SELECT EXISTS (SELECT 1 FROM smart_launches)")):
        raise RuntimeError("Downgrade requires empty ephemeral SMART state")
    op.drop_index("ix_smart_expiry", table_name="smart_launches")
    op.drop_index("ix_idempotency_created", table_name="idempotency_keys")
    op.drop_index("ix_rate_buckets_minute", table_name="rate_buckets")
    op.drop_table("smart_launches")
