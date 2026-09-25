"""Select a cosine acceptance floor on development cases only."""

import json
from pathlib import Path

import embeddings.retrieve as retrieval_module
from backend.evidence_models import RetrievalRequest
from embeddings.retrieve import EvidenceRetriever

ROOT = Path(__file__).resolve().parents[1]


def main():
    cases = [
        row
        for line in (ROOT / "eval/retrieval_cases.jsonl").read_text(encoding="utf-8").splitlines()
        if (row := json.loads(line))["split"] == "development"
    ]
    retrieval_module.MIN_COSINE = -1.0
    retriever = EvidenceRetriever(use_reranker=False)
    rows = []
    for case in cases:
        result = retriever.retrieve(
            RetrievalRequest(
                query=case["query"],
                intent=case["intent"],
                jurisdiction=case.get("jurisdiction"),
                document_ids=case.get("document_ids", []),
                top_k=5,
            )
        )
        # Highest eligible dense similarity is tracked even when BM25/RRF puts
        # a different item first. No candidates means a metadata abstention.
        cosine = result.diagnostics.top_cosine_similarity or -1
        rows.append(
            {
                "case_id": case["case_id"],
                "should_abstain": case["should_abstain"],
                "top_final_cosine": cosine,
                "accepted": result.accepted,
                "document_ids": [item.unit.document_id for item in result.evidence],
            }
        )
    options = []
    for threshold in [round(i / 100, 2) for i in range(20, 66, 2)]:
        false_answers = sum(r["should_abstain"] and r["top_final_cosine"] >= threshold for r in rows)
        answered = sum(not r["should_abstain"] and r["top_final_cosine"] >= threshold for r in rows)
        options.append({"threshold": threshold, "false_answers": false_answers, "answerable_answered": answered})
    safe = [o for o in options if o["false_answers"] == 0]
    max_negative = max(r["top_final_cosine"] for r in rows if r["should_abstain"])
    min_positive = min(r["top_final_cosine"] for r in rows if not r["should_abstain"])
    selected = (
        max(
            safe,
            key=lambda o: (o["answerable_answered"], min(o["threshold"] - max_negative, min_positive - o["threshold"])),
        )
        if safe
        else None
    )
    report = {
        "schema_version": 1,
        "split": "development",
        "cases": rows,
        "candidate_thresholds": options,
        "selected": selected,
    }
    (ROOT / "eval/calibration_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
