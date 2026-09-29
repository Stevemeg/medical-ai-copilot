"""Report selected official ABDM profile compatibility of synthetic fixtures."""

import json
from pathlib import Path
from backend.abdm import analyze


def main():
    rows = []
    for path in sorted(Path("data/synthetic_fhir").glob("syn_pat_*.json")):
        bundle = json.loads(path.read_text(encoding="utf-8"))
        for entry in bundle["entry"]:
            resource = entry["resource"]
            rows.append(
                {"fixture": path.name, "resource_id": resource["id"], **analyze(resource, resource["resourceType"])}
            )
    report = {
        "package_id": "ndhm.in",
        "version": "6.5.0",
        "synthetic_only": True,
        "full_profile_validation": False,
        "results": rows,
    }
    Path("eval/abdm_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"resources_analyzed": len(rows), "full_profile_validation": False}))


if __name__ == "__main__":
    main()
