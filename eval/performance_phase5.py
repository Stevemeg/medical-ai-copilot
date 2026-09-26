"""Local warm CPU candidate-pool evaluation; development split selects configuration."""

import json
from pathlib import Path

from eval.benchmark import cases, metrics, summarize
from embeddings.retrieve import EvidenceRetriever


def main():
    retriever = EvidenceRetriever()
    report = {"development": {}, "held_out": {}}
    for size in (30, 20, 15, 10):
        retriever.rerank_pool_size = size
        rows, latency = summarize(cases("development"), retriever)
        report["development"][str(size)] = {"quality": metrics(rows), "latency": latency}
        print("development", size, report["development"][str(size)], flush=True)
    # The development split has document labels but no recommendation-level
    # labels. Tied document metrics do not justify reducing the established pool.
    selected = 30
    report["selection_reason"] = (
        "Development document metrics tied, so retained the established 30-candidate pool before held-out regression checks."
    )
    retriever.rerank_pool_size = selected
    rows, latency = summarize(cases("held_out"), retriever)
    report["selected_pool"] = selected
    report["held_out"] = {"quality": metrics(rows), "latency": latency}
    Path("eval/performance_phase5_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("selected", selected, report["held_out"], flush=True)


if __name__ == "__main__":
    main()
