"""Print a synthetic patient-view request for local authenticated demonstration."""

import argparse
import json
from pathlib import Path
from uuid import uuid4

from backend.fhir_client import RESOURCE_TYPES


def request_body(number: int = 2) -> dict:
    if number not in range(1, 6):
        raise ValueError("Select a bundled synthetic patient 1-5")
    bundle = json.loads(Path(f"data/synthetic_fhir/syn_pat_{number:03}.json").read_text(encoding="utf-8"))
    resources = [e["resource"] for e in bundle["entry"]]
    patient = next(r for r in resources if r["resourceType"] == "Patient")
    prefetch = {
        "patient": patient,
        **{
            kind: {
                "resourceType": "Bundle",
                "type": "searchset",
                "entry": [{"resource": r} for r in resources if r["resourceType"] == kind],
            }
            for kind in RESOURCE_TYPES
        },
    }
    return {
        "hook": "patient-view",
        "hookInstance": str(uuid4()),
        "context": {"userId": "Practitioner/synthetic-demo", "patientId": patient["id"]},
        "prefetch": prefetch,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patient", type=int, default=2)
    print(json.dumps(request_body(parser.parse_args().patient), indent=2))
