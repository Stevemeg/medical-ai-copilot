"""Offline end-to-end safety scenarios using real retrieval and NLI, fake LLM."""

import json
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np

from backend.evidence_models import RetrievalRequest
from backend.evidence_pipeline import EvidenceAnswerService
from embeddings.retrieve import EvidenceRetriever

ROOT = Path(__file__).resolve().parents[1]
REC_ID = "nice-ng136-2026-02-26:rec:1.4.24"


class FixedGenerator:
    def generate(self, system_prompt, user_prompt):
        return json.dumps(
            {
                "status": "grounded",
                "claims": [
                    {
                        "claim_id": "c1",
                        "text": "Provide an annual review of care for adults with hypertension to monitor blood pressure.",
                        "evidence_ids": [REC_ID],
                    }
                ],
            }
        )


def main():
    retriever = EvidenceRetriever(use_reranker=True)
    service = EvidenceAnswerService(retriever, FixedGenerator())
    queries = {
        "supported_current": RetrievalRequest(query="What ongoing review is advised for adults who have hypertension?"),
        "out_of_domain": RetrievalRequest(query="How do I repair a motorcycle engine?"),
        "stale_current": RetrievalRequest(query="What did old NG19 guidance say?", document_ids=["nice-ng19"]),
        "stale_historical": RetrievalRequest(
            query="What did the previous diabetic foot guidance say about SINBAD classification?",
            intent="historical",
            document_ids=["nice-ng19"],
        ),
        "exact_term": RetrievalRequest(query="What is the role of QRISK3 in cardiovascular prevention?"),
        "reference": RetrievalRequest(query="Explain how skeletal muscles contract", intent="reference_explanation"),
    }
    report = {}
    for label, request in queries.items():
        started = perf_counter()
        retrieval = retriever.retrieve(request)
        answer = service.answer(request) if label in ("supported_current", "out_of_domain") else None
        report[label] = {
            "retrieval_accepted": retrieval.accepted,
            "reranker_used": retrieval.diagnostics.reranker_used,
            "top_evidence_ids": [item.unit.evidence_unit_id for item in retrieval.evidence],
            "top_lifecycle": [item.unit.lifecycle_status.value for item in retrieval.evidence],
            "dense_count": retrieval.diagnostics.dense_count,
            "bm25_count": retrieval.diagnostics.bm25_count,
            "answer_status": answer.status.value if answer else None,
            "claim_support": [claim.support_status.value for claim in answer.claims] if answer else [],
            "full_pipeline_ms": (perf_counter() - started) * 1000,
        }
    # Warm, repeated pipeline timing with model-backed verification and a
    # bounded evidence set. No provider network is involved.
    times = []
    for _ in range(7):
        started = perf_counter()
        answer = service.answer(
            RetrievalRequest(
                query="What does recommendation 1.4.24 say about annual care review?", recommendation_ids=["1.4.24"]
            )
        )
        times.append((perf_counter() - started) * 1000)
    report["pipeline_latency"] = {
        "median_ms": median(times),
        "p95_ms": float(np.percentile(times, 95)),
        "samples": len(times),
        "last_status": answer.status.value,
    }
    assert report["supported_current"]["answer_status"] == "grounded"
    assert report["supported_current"]["claim_support"] == ["supported"]
    assert report["out_of_domain"]["answer_status"] == "abstained"
    assert not report["stale_current"]["retrieval_accepted"]
    assert report["stale_historical"]["retrieval_accepted"]
    assert set(report["stale_historical"]["top_lifecycle"]) == {"superseded"}
    assert report["exact_term"]["retrieval_accepted"] and report["exact_term"]["bm25_count"] > 0
    assert report["reference"]["retrieval_accepted"]
    assert report["pipeline_latency"]["last_status"] == "grounded"
    (ROOT / "eval/scenario_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
