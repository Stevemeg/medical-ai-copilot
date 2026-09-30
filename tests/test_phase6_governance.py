"""Conservative diff matching, release gates and boundary invariants."""

import json
from pathlib import Path

import pytest

from backend.guideline_updates import CandidateInput, recommendation_diff
from eval.run_all import gate, injections, load, safety


def rec(ident, text, section="one"):
    return {"recommendation_id": ident, "section": section, "text": text}


def test_diff_all_classifications_and_text_record():
    old = [
        rec("1.1.1", "Keep this paragraph unchanged."),
        rec("1.1.2", "The marker is blue."),
        rec("1.1.3", "Remove this paragraph."),
    ]
    new = [
        rec("1.1.1", "Keep   this paragraph unchanged."),
        rec("1.1.2", "The marker is green."),
        rec("1.1.4", "Add this paragraph."),
    ]
    result = recommendation_diff(old, new)
    assert all(len(result[key]) == 1 for key in ("added", "removed", "modified", "unchanged"))
    assert "-The marker is blue." in result["modified"][0]["text_diff"]
    assert "+The marker is green." in result["modified"][0]["text_diff"]


@pytest.mark.parametrize(
    "old,new",
    [
        ([rec("1.1.1", "A"), rec("1.1.1", "B")], [rec("1.1.1", "C")]),
        ([rec(None, "A")], [rec(None, "B")]),
        ([rec(None, "A", "")], [rec(None, "A", "")]),
    ],
)
def test_ambiguous_matching_is_not_forced(old, new):
    result = recommendation_diff(old, new)
    assert result["ambiguous"] and not result["modified"]


def test_candidate_contract_rejects_unbounded_and_invalid_metadata():
    base = dict(
        document_id="test",
        version_id="v1",
        source_url="https://example.org",
        updated_at="2026-02-30",
        license_note="Synthetic test content only",
        recommendations=[{"section": "one", "text": "Synthetic text with sufficient length.", "page": 1}],
    )
    with pytest.raises(ValueError):
        CandidateInput(**base)
    base["updated_at"] = "2026-02-28"
    base["recommendations"][0]["text"] = "a" * 2501
    with pytest.raises(ValueError):
        CandidateInput(**base)


def test_evaluation_gate_cannot_silently_accept_missing_or_regressed_metrics():
    baseline = {"gates": {"metric.value": {"min": 0.9}, "safety.false_answers": {"max": 0}}}
    assert len(gate({}, baseline)) == 2
    assert len(gate({"metric": {"value": 0.8}, "safety": {"false_answers": 1}}, baseline)) == 2
    assert gate({"metric": {"value": 0.95}, "safety": {"false_answers": 0}}, baseline) == []


def test_versioned_evaluation_provenance_and_split_separation():
    for suite in ("retrieval", "grounding", "clinical_rules", "prompt_injection"):
        cases = load(suite)
        assert {c["split"] for c in cases} == {"development", "held_out"}
    cases = load("retrieval")
    dev = {c["query"] for c in cases if c["split"] == "development"}
    held = {c["query"] for c in cases if c["split"] == "held_out"}
    assert not dev & held
    assert len(cases) >= 40
    assert len(load("grounding")) >= 30


def test_final_output_injection_and_failure_safety_contracts():
    assert injections(load("prompt_injection"))["pass_rate"] == 1
    assert safety()["pass_rate"] == 1


def test_governed_grounding_passages_resolve_to_exact_current_units():
    units = {u["evidence_unit_id"]: u for u in json.loads(Path("data/evidence_units.json").read_text(encoding="utf-8"))}
    cases = [c for c in load("grounding") if "governed_unit" in c]
    identifiers = set()
    for case in cases:
        supplied = case["governed_unit"]
        assert units[supplied["evidence_unit_id"]] == supplied
        assert supplied["lifecycle_status"] == "current"
        identifiers.add(supplied["evidence_unit_id"])
    assert len(identifiers) == len(cases) == 16
    assert {c["split"] for c in cases} == {"development", "held_out"}


def test_no_preview_abdm_profiles():
    manifest = json.loads(Path("data/abdm/manifest.json").read_text())
    assert manifest["version"] == "6.5.0"
    assert manifest["fhir_version"] == "4.0.1"
