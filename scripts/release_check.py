"""Non-destructive release provenance validation. No migrations or online updates."""

import argparse
import hashlib
import json
import platform
import subprocess  # nosec B404 # Reviewed: fixed read-only Git commands, no shell.
import shutil
from pathlib import Path

from backend.abdm import profiles, validate_recorded_example
from backend.index_provenance import validate_chunks_manifest, validate_manifest
from backend.rule_evidence import RecommendationEvidenceRegistry
from backend.source_registry import SourceRegistry, sha256_file

ROOT = Path(__file__).resolve().parents[1]
GENERATED = {"eval/final_system_report.json", "release_manifest.json", "PHASE6_FINAL_REPORT.md"}
GIT = shutil.which("git")
if GIT is None:
    raise RuntimeError("Git is required for release verification")


def source_digest() -> str:
    paths = subprocess.check_output([GIT, "ls-files"], cwd=ROOT, text=True).splitlines()  # nosec B603 # Fixed local Git command.
    digest = hashlib.sha256()
    for relative in sorted(set(paths) - GENERATED):
        path = ROOT / relative
        if path.is_file():
            # Normalize text endings across Windows and Linux checkouts.
            content = path.read_bytes()
            if b"\x00" not in content:
                content = content.replace(b"\r\n", b"\n")
            digest.update(relative.encode() + b"\x00" + hashlib.sha256(content).digest())
    return digest.hexdigest()


def manifest() -> dict:
    return {
        "schema_version": 1,
        "git_sha": subprocess.check_output([GIT, "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),  # nosec B603 # Fixed Git command.
        "source_digest": source_digest(),
        "schema_revision": "62a2guidelines",
        "knowledge_registry_sha256": sha256_file(ROOT / "data/source_registry.json"),
        "rule_evidence_registry_sha256": sha256_file(ROOT / "data/rule_evidence_registry.json"),
        "vector_manifests": {
            label: sha256_file(ROOT / f"data/vector_store/{label}_faiss.manifest.json")
            for label in ("clinical", "anatomy")
        },
        "evaluation_baseline_version": "1.0.0",
        "evaluation_report_sha256": sha256_file(ROOT / "eval/final_system_report.json"),
        "model_manifest_sha256": sha256_file(ROOT / "model_manifest.json"),
        "abdm_manifest_sha256": sha256_file(ROOT / "data/abdm/manifest.json"),
        "fhir_version": "4.0.1",
        "smart_version": "2.2.0",
        "cds_hooks_version": "2.0.1",
        "patient_view_version": "1.0",
        "abdm_ig_version": "6.5.0",
        "python_version": platform.python_version(),
    }


def validate_attestation(recorded: dict, report: dict, expected: dict) -> None:
    if not report["passed"] or report["git"]["dirty"] or report["mode"] != "local_models":
        raise ValueError("Clean, passing model evaluation required")
    for key, value in expected.items():
        if key not in {"git_sha", "python_version"} and recorded.get(key) != value:
            raise ValueError("Release metadata mismatch: " + key)
    if recorded["git_sha"] != report["git"]["sha"]:
        raise ValueError("Manifest and evaluated code SHA differ")
    if recorded["python_version"] != report["environment"]["python"]:
        raise ValueError("Manifest and evaluation environment differ")


def check(release: bool = False) -> dict:
    registry = SourceRegistry()
    RecommendationEvidenceRegistry()
    validate_chunks_manifest(ROOT / "data/chunks.json", registry)
    for label in ("clinical", "anatomy"):
        validate_manifest(
            ROOT / f"data/vector_store/{label}_faiss.index", ROOT / f"data/vector_store/{label}_metadata.json", registry
        )
    profiles()
    validate_recorded_example()
    report_path = ROOT / "eval/final_system_report.json"
    if release:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        recorded = json.loads((ROOT / "release_manifest.json").read_text(encoding="utf-8"))
        validate_attestation(recorded, report, manifest())
        import re

        if not re.fullmatch(r"[0-9a-f]{40}", recorded["git_sha"]):
            raise ValueError("Invalid evaluated SHA")
        subprocess.run([GIT, "merge-base", "--is-ancestor", recorded["git_sha"], "HEAD"], cwd=ROOT, check=True)  # nosec B603 # Validated SHA, fixed command.
    return {"provenance": "passed", "release_report_checked": release}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", action="store_true")
    parser.add_argument("--write-manifest", action="store_true")
    args = parser.parse_args()
    if args.write_manifest:
        (ROOT / "release_manifest.json").write_text(json.dumps(manifest(), indent=2) + "\n", encoding="utf-8")
    print(json.dumps(check(args.release), indent=2))


if __name__ == "__main__":
    main()
