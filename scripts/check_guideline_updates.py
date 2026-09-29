"""Safely check registered structured sources; never activate new content."""

import argparse
import hashlib
import json
from pathlib import Path

from backend.guideline_updates import CandidateInput, acquire_registered


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/guideline_monitoring.json"))
    args = parser.parse_args()
    rows = json.loads(args.manifest.read_text(encoding="utf-8"))
    report = []
    for source in rows["sources"]:
        if source["mode"] == "manual":
            report.append({"document_id": source["document_id"], "status": "manual_registration_required"})
            continue
        try:
            raw = acquire_registered(source["document_id"], source["url"])
            candidate = CandidateInput.model_validate_json(raw)
            if candidate.document_id != source["document_id"]:
                raise ValueError("Source identity mismatch")
            checksum = hashlib.sha256(raw).hexdigest()
            report.append(
                {
                    "document_id": candidate.document_id,
                    "version_id": candidate.version_id,
                    "checksum": checksum,
                    "status": "changed" if checksum != source.get("checksum") else "unchanged",
                }
            )
        except (ValueError, OSError):
            report.append({"document_id": source["document_id"], "status": "failed"})
    print(json.dumps({"auto_activation": False, "sources": report}, indent=2))
    return int(any(row["status"] == "failed" for row in report))


if __name__ == "__main__":
    raise SystemExit(main())
