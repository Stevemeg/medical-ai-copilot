"""Local synthetic claim/evidence benchmark for verifier and citation survival."""

import json
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np

from backend.evidence_models import AnswerClaim, EvidenceUnit, SupportStatus
from backend.grounding import NLIClaimVerifier

ROOT = Path(__file__).resolve().parents[1]


def unit(ident, text):
    return EvidenceUnit(
        evidence_unit_id=ident,
        unit_type="context_chunk",
        chunk_id=ident,
        source_chunk_ids=[ident],
        document_id="synthetic",
        version_id="synthetic-v1",
        text=text,
        source_type="clinical_guideline",
        jurisdiction="test",
        lifecycle_status="current",
        publisher="Synthetic",
        canonical_title="Synthetic evidence",
        page_start=1,
        page_end=1,
    )


def evaluate(cases, verifier):
    rows = []
    for case in cases:
        evidence = [unit(ident, text) for ident, text in case["evidence"].items() if ident in case["cited_ids"]]
        claim = AnswerClaim(claim_id=case["case_id"], text=case["claim"], evidence_ids=case["cited_ids"])
        started = perf_counter()
        status = verifier.verify(claim, evidence)
        elapsed = (perf_counter() - started) * 1000
        rows.append({**case, "actual": status.value, "latency_ms": elapsed})
    supported = [r for r in rows if r["expected"] == "supported"]
    unsupported = [r for r in rows if r["expected"] != "supported"]
    predicted = [r for r in rows if r["actual"] == SupportStatus.SUPPORTED.value]
    # Case-qualified IDs prevent unrelated cases with the same short fixture
    # identifier from being counted as the same citation.
    citation_pairs = [(r["case_id"], ident) for r in predicted for ident in r["cited_ids"]]
    relevant_pairs = {(r["case_id"], ident) for r in supported for ident in r["relevant_ids"]}
    correct = len(set(citation_pairs) & relevant_pairs)
    metrics = {
        "supported_claim_precision": sum(r["expected"] == "supported" for r in predicted) / len(predicted)
        if predicted
        else None,
        "unsupported_claim_detection_rate": sum(r["actual"] != "supported" for r in unsupported) / len(unsupported),
        "false_supported_rate": sum(r["actual"] == "supported" for r in unsupported) / len(unsupported),
        "citation_precision": correct / len(citation_pairs) if citation_pairs else None,
        "citation_recall": correct / len(relevant_pairs) if relevant_pairs else None,
        "verification_median_ms": median(r["latency_ms"] for r in rows),
        "verification_p95_ms": float(np.percentile([r["latency_ms"] for r in rows], 95)),
        "support_cases": len(supported),
        "nonsupport_cases": len(unsupported),
        "predicted_supported": len(predicted),
    }
    return {"metrics": metrics, "results": rows}


def main():
    cases = [
        json.loads(line) for line in (ROOT / "eval/grounding_cases.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    verifier = NLIClaimVerifier()
    report = {
        "schema_version": 1,
        "development": evaluate([c for c in cases if c["split"] == "development"], verifier),
        "held_out": evaluate([c for c in cases if c["split"] == "held_out"], verifier),
    }
    (ROOT / "eval/grounding_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key]["metrics"] for key in ("development", "held_out")}, indent=2))


if __name__ == "__main__":
    main()
