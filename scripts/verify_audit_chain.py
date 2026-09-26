"""Run with python -m scripts.verify_audit_chain."""

from backend.db import session_factory
from backend.durable_audit import verify_chain


def main() -> int:
    with session_factory()() as session:
        result = verify_chain(session)
    print(result["message"])
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
