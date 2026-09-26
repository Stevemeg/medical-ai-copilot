"""Postgres backed request throttling and minimized operational audit."""

from datetime import datetime, timezone

from sqlalchemy.dialects.postgresql import insert

from backend.db import RateBucket, transaction
from backend.durable_audit import append_event
from backend.security import ActorContext
from backend.settings import get_settings


def audit_operation(
    event_type: str,
    actor: ActorContext,
    request_id: str,
    resource_type: str = "request",
    resource_id: str = "-",
    payload: dict | None = None,
) -> None:
    with transaction() as session:
        append_event(
            session,
            event_type=event_type,
            resource_type=resource_type,
            resource_id=resource_id,
            payload=payload,
            actor_subject=actor.subject,
            actor_roles=sorted(actor.roles),
            request_id=request_id,
        )


def rate_limit(actor: ActorContext, operation: str) -> bool:
    minute = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    with transaction() as session:
        stmt = insert(RateBucket).values(actor_subject=actor.subject, operation=operation, minute=minute, count=1)
        upsert = stmt.on_conflict_do_update(
            index_elements=[RateBucket.actor_subject, RateBucket.operation, RateBucket.minute],
            set_={"count": RateBucket.count + 1},
        ).returning(RateBucket.count)
        count = session.scalar(upsert)
        return count is not None and count <= get_settings().rate_limit_per_minute
