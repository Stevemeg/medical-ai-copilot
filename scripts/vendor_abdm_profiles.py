"""Copy exact selected official profiles from a checksum-pinned local IG package."""

import hashlib
import json
import tarfile
from pathlib import Path

PACKAGE_SHA256 = "41fa99955adbd1983a235d7c7136b30f423ce34ff541d86b354d95b7704527d4"
PROFILES = (
    "Patient",
    "Condition",
    "Observation",
    "ObservationVitalSigns",
    "MedicationRequest",
    "AllergyIntolerance",
    "Encounter",
    "Procedure",
)


def vendor(package: Path, destination: Path) -> dict:
    if package.stat().st_size > 30_000_000 or hashlib.sha256(package.read_bytes()).hexdigest() != PACKAGE_SHA256:
        raise ValueError("ABDM package size/checksum mismatch")
    destination.mkdir(parents=True, exist_ok=True)
    hashes = {}
    with tarfile.open(package, "r:gz") as archive:
        for name in PROFILES:
            member = archive.getmember(f"package/StructureDefinition-{name}.json")
            if not member.isfile() or member.size > 3_000_000:
                raise ValueError("Unexpected profile member")
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError("Missing profile")
            raw = stream.read(3_000_001)
            profile = json.loads(raw)
            if profile["version"] != "6.5.0" or profile["fhirVersion"] != "4.0.1":
                raise ValueError("Wrong profile release")
            filename = f"StructureDefinition-{name}.json"
            (destination / filename).write_bytes(raw)
            hashes[filename] = hashlib.sha256(raw).hexdigest()
    manifest = {
        "package_id": "ndhm.in",
        "version": "6.5.0",
        "fhir_version": "4.0.1",
        "license": "CC0-1.0",
        "official_url": "https://www.nrces.in/ndhm/fhir/r4/package.tgz",
        "verified_on": "2026-09-29",
        "package_sha256": PACKAGE_SHA256,
        "profiles": hashes,
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    print(json.dumps(vendor(Path("reports/abdm-package.tgz"), Path("data/abdm")), indent=2))
