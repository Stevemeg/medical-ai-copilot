"""Audit installed dependencies; only exact, unexpired reviewed exceptions pass."""

import json
import re
import subprocess  # nosec B404 #- reviewed: fixed local interpreter/module, no shell or user arguments
import sys
from datetime import date
from pathlib import Path
from importlib.metadata import distributions


def audit_version(name: str, installed: str) -> str:
    # The official CPU wheel shares the reviewed upstream code release. Keep
    # the exact mapping explicit; other unknown local builds still fail audit.
    return "2.14.0" if name == "torch" and installed == "2.14.0+cpu" else installed


def package_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def main():
    output = Path("reports/pip-audit.json")
    output.parent.mkdir(exist_ok=True)
    inventory = sorted((package_name(d.metadata["Name"]), d.version) for d in distributions())
    requirements = output.parent / "audit-installed.txt"
    requirements.write_text("".join(f"{name}=={audit_version(name, installed)}\n" for name, installed in inventory))
    mappings = [
        {"package": name, "installed": installed, "audited_upstream": audit_version(name, installed)}
        for name, installed in inventory
        if audit_version(name, installed) != installed
    ]
    output.unlink(missing_ok=True)
    completed = subprocess.run(  # nosec B603 #- fixed pip_audit command and local report path only
        [
            sys.executable,
            "-m",
            "pip_audit",
            "--no-deps",
            "--disable-pip",
            "-r",
            str(requirements),
            "--format",
            "json",
            "--output",
            str(output),
        ],
        check=False,
    )
    if completed.returncode not in (0, 1) or not output.exists():
        return 2
    result = json.loads(output.read_text(encoding="utf-8"))
    exceptions = json.loads(Path("security_exceptions.json").read_text())["exceptions"]
    unresolved, contained = [], []
    scanned = {package_name(d["name"]) for d in result["dependencies"]}
    unresolved.extend(
        {"package": name, "reason": "Installed distribution absent from scanner report"}
        for name, _ in inventory
        if name not in scanned
    )
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
    summary = {
        "unresolved": unresolved,
        "contained": contained,
        "raw_report": str(output),
        "audited_distributions": len(inventory),
        "build_version_mappings": mappings,
    }
    Path("reports/security-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return int(bool(unresolved))


if __name__ == "__main__":
    raise SystemExit(main())
