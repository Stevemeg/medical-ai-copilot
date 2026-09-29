"""Run versioned contract cases against real boundary tests, reporting every skip."""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from xml.etree import ElementTree

ROOT = Path(__file__).resolve().parents[1]


def run(include_postgres=False):
    cases = [json.loads(line) for line in (ROOT / "eval/cases/v1/contracts.jsonl").read_text().splitlines()]
    selected = [c for c in cases if include_postgres or "postgres" not in c["tags"]]
    if include_postgres and not os.environ.get("PHASE5_TEST_DATABASE_URL"):
        raise ValueError("PostgreSQL contract evaluation requires PHASE5_TEST_DATABASE_URL")
    nodes = [c["input"]["pytest_node"] for c in selected]
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "contracts.xml"
        completed = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", *nodes, "--junitxml=" + str(output)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if not output.exists():
            raise RuntimeError("Contract execution did not produce results: " + completed.stdout[-2000:])
        executed = list(ElementTree.parse(output).iter("testcase"))
    results = []
    for case in selected:
        file, function = case["input"]["pytest_node"].split("::")
        classname = file[:-3].replace("/", ".")
        rows = [r for r in executed if r.get("classname") == classname and r.get("name", "").split("[")[0] == function]
        failures = sum(r.find("failure") is not None or r.find("error") is not None for r in rows)
        skips = sum(r.find("skipped") is not None for r in rows)
        results.append(
            {
                "case_id": case["case_id"],
                "suite": case["suite"],
                "split": case["split"],
                "scenario_count": len(rows),
                "failed": failures,
                "skipped": skips,
                "passed": bool(rows) and not failures and not skips,
                "duration_seconds": sum(float(r.get("time", "0")) for r in rows),
            }
        )
    suites = {}
    for suite in sorted({r["suite"] for r in results}):
        rows = [r for r in results if r["suite"] == suite]
        suites[suite] = {
            "cases": len(rows),
            "scenarios": sum(r["scenario_count"] for r in rows),
            "passed": all(r["passed"] for r in rows),
        }
    return {
        "cases": len(results),
        "scenarios": len(executed),
        "pass_rate": sum(r["passed"] for r in results) / len(results),
        "postgres_included": include_postgres,
        "suites": suites,
        "results": results,
        "process_exit_code": completed.returncode,
    }
