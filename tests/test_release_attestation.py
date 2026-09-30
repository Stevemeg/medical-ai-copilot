"""A source digest alone must not allow forged release artifact metadata."""

from copy import deepcopy

import pytest

from scripts.release_check import validate_attestation


@pytest.mark.parametrize(
    "field",
    [
        "source_digest",
        "evaluation_report_sha256",
        "knowledge_registry_sha256",
        "rule_evidence_registry_sha256",
        "vector_manifests",
        "schema_revision",
        "smart_version",
        "abdm_ig_version",
    ],
)
def test_release_rejects_metadata_and_report_tampering(field):
    expected = {"git_sha": "a" * 40, "python_version": "3.11.0", field: "verified-value"}
    report = {
        "passed": True,
        "git": {"dirty": False, "sha": "a" * 40},
        "mode": "local_models",
        "environment": {"python": "3.11.0"},
    }
    validate_attestation(expected, report, expected)
    tampered = deepcopy(expected)
    tampered[field] = "changed-value"
    with pytest.raises(ValueError, match=field):
        validate_attestation(tampered, report, expected)


@pytest.mark.parametrize("change", ["dirty", "failed", "fake_models", "wrong_sha", "wrong_python"])
def test_release_requires_clean_matching_model_evaluation(change):
    recorded = {"git_sha": "a" * 40, "python_version": "3.11.0"}
    report = {
        "passed": True,
        "git": {"dirty": change == "dirty", "sha": "b" * 40 if change == "wrong_sha" else "a" * 40},
        "mode": "deterministic_contracts" if change == "fake_models" else "local_models",
        "environment": {"python": "3.12.0" if change == "wrong_python" else "3.11.0"},
    }
    report["passed"] = change != "failed"
    with pytest.raises(ValueError):
        validate_attestation(recorded, report, recorded)
