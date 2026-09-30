"""Generate and mask ephemeral test keys in a GitHub Actions job."""

import os
import secrets
from pathlib import Path


def main():
    target = Path(os.environ["GITHUB_ENV"])
    with target.open("a", encoding="utf-8") as output:
        for name in ("AUDIT_HMAC_KEY", "DEV_JWT_KEY"):
            value = secrets.token_hex(32)
            print("::add-mask::" + value)
            output.write(name + "=" + value + "\n")


if __name__ == "__main__":
    main()
