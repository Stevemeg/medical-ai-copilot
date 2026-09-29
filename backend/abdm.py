"""Selected official profile analysis; explicitly not full FHIR validation."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "data/abdm"


def validate_recorded_example() -> None:
    root = ROOT.parents[1]
    record = json.loads((root / "eval/abdm_validation_manifest.json").read_text())
    if record["package_version"] != "6.5.0":
        raise ValueError("Recorded validation package drift")
    for result in record["results"]:
        for kind in ("example", "outcome"):
            if hashlib.sha256((root / result[kind]).read_bytes()).hexdigest() != result[kind + "_sha256"]:
                raise ValueError("Recorded validator input/output drift; repeat validation")


def profiles() -> dict[str, dict]:
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    if manifest["package_id"] != "ndhm.in" or manifest["version"] != "6.5.0":
        raise ValueError("Unsupported ABDM package")
    result = {}
    for filename, digest in manifest["profiles"].items():
        raw = (ROOT / filename).read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError("Official profile integrity mismatch")
        profile = json.loads(raw)
        result[profile["id"]] = profile
    return result


def analyze(resource: dict, profile_name: str) -> dict:
    profile = profiles()[profile_name]
    if resource.get("resourceType") != profile["type"]:
        raise ValueError("Resource/profile type mismatch")
    required, missing, bindings = [], [], []
    # Inspect direct required fields from the official snapshot. Complex
    # slicing, terminology and FHIRPath invariants require the HL7 validator.
    for element in profile["snapshot"]["element"]:
        path = element["path"]
        if element.get("binding"):
            bindings.append({"path": path, **element["binding"]})
        if path.count(".") == 1 and element.get("min", 0) > 0 and ":" not in element["id"]:
            key = path.split(".")[1]
            required.append(key)
            present = (
                any(k.startswith(key[:-3]) and v is not None for k, v in resource.items())
                if key.endswith("[x]")
                else bool(resource.get(key))
            )
            if not present:
                missing.append(key)
    return {
        "profile": profile["url"],
        "version": profile["version"],
        "required_direct_fields": required,
        "missing_required_direct_fields": missing,
        "terminology_bindings": bindings,
        "validation_scope": "official snapshot direct cardinality analysis; no full conformance assertion",
        "unvalidated": [
            "nested cardinality",
            "slicing",
            "FHIRPath invariants",
            "terminology membership",
            "reference target profiles",
        ],
    }
