"""Final held-out regression measurement after evidence-level correction."""

import json
from pathlib import Path

from eval.benchmark import cases, metrics, summarize
from embeddings.retrieve import EvidenceRetriever, RERANK_POOL_SIZE


def main():
    retriever = EvidenceRetriever()
    summarize(cases("development"), retriever)  # warm both embedding and reranker models
    rows, latency = summarize(cases("held_out"), retriever)
    phase4 = json.loads(Path("eval/retrieval_report.json").read_text(encoding="utf-8"))["held_out"]
    report = {
        "selected_pool": RERANK_POOL_SIZE,
        "selection_reason": "Development document metrics tied; smaller pools failed held-out evidence-unit regression, so the established 30-unit pool was retained.",
        "phase4": {"quality": phase4["phase4"], "latency": phase4["latency"]},
        "phase5": {"quality": metrics(rows), "latency": latency},
    }
    Path("eval/final_phase5_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
