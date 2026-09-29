"""Report expired operational rows; deletion requires --execute."""

import argparse
import json

from backend.db import transaction
from backend.retention import cleanup
from backend.settings import get_settings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    with transaction() as session:
        counts = cleanup(session, get_settings(), execute=args.execute)
    print(json.dumps({"mode": "execute" if args.execute else "dry_run", "counts": counts}, indent=2))


if __name__ == "__main__":
    main()
