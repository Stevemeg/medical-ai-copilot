"""Bounded cleanup of explicitly ephemeral operational state only."""

from datetime import timedelta

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from backend.db import IdempotencyKey, RateBucket, SmartLaunch
from backend.settings import Settings


def cleanup(session: Session, settings: Settings, *, execute: bool = False) -> dict[str, int]:
    # Use database transaction time so application clock skew cannot erase an
    # active minute. Strict inequality preserves the complete retry window.
    now = session.scalar(select(func.now()))
    if now is None:
        raise RuntimeError("Database clock unavailable")
    predicates = (
        (RateBucket, RateBucket.minute < now - timedelta(hours=settings.rate_bucket_retention_hours)),
        (IdempotencyKey, IdempotencyKey.created_at < now - timedelta(days=settings.idempotency_retention_days)),
    )
    counts = {}
    for model, predicate in predicates:
        if execute:
            # Skip rows in use by another transaction. There is no DDL, table
            # truncation or access to clinical/history tables in this routine.
            ids = select(model.id).where(predicate).with_for_update(skip_locked=True)
            result = session.execute(delete(model).where(model.id.in_(ids)).returning(model.id))
            counts[model.__tablename__] = len(result.all())
        else:
            counts[model.__tablename__] = session.scalar(select(func.count()).select_from(model).where(predicate)) or 0
    expired = SmartLaunch.expires_at < now
    if execute:
        handles = select(SmartLaunch.state_hash).where(expired).with_for_update(skip_locked=True)
        counts["smart_launches"] = len(
            session.execute(
                delete(SmartLaunch).where(SmartLaunch.state_hash.in_(handles)).returning(SmartLaunch.state_hash)
            ).all()
        )
    else:
        counts["smart_launches"] = session.scalar(select(func.count()).select_from(SmartLaunch).where(expired)) or 0
    return counts
