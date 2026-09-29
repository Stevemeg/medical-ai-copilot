"""Audit installed dependencies; only exact, unexpired reviewed exceptions pass."""

import json
import subprocess  # nosec B404 #- reviewed: fixed local interpreter/module, no shell or user arguments
import sys
from datetime import date
from pathlib import Path


def main():
    output = Path("reports/pip-audit.json")
    output.parent.mkdir(exist_ok=True)
    completed = subprocess.run(  # nosec B603 #- fixed pip_audit command and local report path only
        [sys.executable, "-m", "pip_audit", "--format", "json", "--output", str(output)], check=False
    )
    if completed.returncode not in (0, 1) or not output.exists():
        return 2
    result = json.loads(output.read_text(encoding="utf-8"))
    exceptions = json.loads(Path("security_exceptions.json").read_text())["exceptions"]
    unresolved, contained = [], []
    for package in result["dependencies"]:
        if package.get("skip_reason"):
            unresolved.append({"package": package["name"], "reason": "Dependency was skipped by scanner"})
        for advisory in package.get("vulns", []):
            match = next(
                (
                    e
                    for e in exceptions
                    if e["package"] == package["name"]
                    and e["version"] == package["version"]
                    and e["advisory"] == advisory["id"]
                    and date.today() <= date.fromisoformat(e["review_by"])
                ),
                None,
            )
            record = {"package": package["name"], "version": package["version"], "advisory": advisory["id"]}
            (contained if match else unresolved).append(record)
    summary = {"unresolved": unresolved, "contained": contained, "raw_report": str(output)}
    Path("reports/security-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return int(bool(unresolved))


if __name__ == "__main__":
    raise SystemExit(main())
