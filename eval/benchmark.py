"""Reproducible local document-level retrieval benchmark; no API calls."""

import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path
from statistics import median

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from backend.evidence_models import RetrievalRequest
from backend.knowledge_models import RetrievalPolicy
from backend.source_registry import SourceRegistry
from embeddings.retrieve import EvidenceRetriever, simple_tokenize

ROOT = Path(__file__).resolve().parents[1]
BASELINE_SHA = "c35bdf82bfdcc1ace686b190cc577f3f9ddba5c7"


def cases(split):
    return [
        row
        for line in (ROOT / "eval/retrieval_cases.jsonl").read_text(encoding="utf-8").splitlines()
        if (row := json.loads(line))["split"] == split
    ]


def percentile(values, p):
    if not values:
        return None
    return float(np.percentile(values, p))


def metrics(rows):
    rows = [{**row, "ranked_documents": list(dict.fromkeys(row["ranked_documents"]))} for row in rows]
    answerable = [r for r in rows if not r["should_abstain"]]
    unanswerable = [r for r in rows if r["should_abstain"]]

    def rr(row):
        return next(
            (1 / (i + 1) for i, doc in enumerate(row["ranked_documents"]) if doc in row["relevant_document_ids"]), 0
        )

    def recall(row, k):
        return len(set(row["ranked_documents"][:k]) & set(row["relevant_document_ids"])) / len(
            row["relevant_document_ids"]
        )

    def ndcg(row):
        hits = [
            1 / np.log2(i + 2)
            for i, doc in enumerate(row["ranked_documents"][:10])
            if doc in row["relevant_document_ids"]
        ]
        ideal = sum(1 / np.log2(i + 2) for i in range(min(10, len(row["relevant_document_ids"]))))
        return sum(hits) / ideal if ideal else 0

    labeled = [r for r in rows if r.get("relevant_evidence_ids")]
    evidence_recall = (
        float(
            np.mean(
                [
                    len(set(r.get("ranked_ids", [])[:5]) & set(r["relevant_evidence_ids"]))
                    / len(r["relevant_evidence_ids"])
                    for r in labeled
                ]
            )
        )
        if labeled and any("ranked_ids" in r for r in labeled)
        else None
    )
    evidence_mrr = (
        float(
            np.mean(
                [
                    next(
                        (
                            1 / (i + 1)
                            for i, ident in enumerate(r.get("ranked_ids", []))
                            if ident in r["relevant_evidence_ids"]
                        ),
                        0,
                    )
                    for r in labeled
                ]
            )
        )
        if evidence_recall is not None
        else None
    )
    return {
        "recall_at_5": float(np.mean([recall(r, 5) for r in answerable])),
        "recall_at_10": float(np.mean([recall(r, 10) for r in answerable])),
        "mrr": float(np.mean([rr(r) for r in answerable])),
        "ndcg_at_10": float(np.mean([ndcg(r) for r in answerable])),
        "evidence_recall_at_5_labeled_subset": evidence_recall,
        "evidence_mrr_labeled_subset": evidence_mrr,
        "answerable_answered": sum(bool(r["accepted"]) for r in answerable),
        "answerable_total": len(answerable),
        "unanswerable_abstained": sum(not r["accepted"] for r in unanswerable),
        "unanswerable_total": len(unanswerable),
        "false_answer_rate": sum(bool(r["accepted"]) for r in unanswerable) / len(unanswerable),
    }


def summarize(cases, retriever):
    rows = []
    latencies = {name: [] for name in ("dense", "bm25", "reranking", "full_retrieval")}
    for case in cases:
        request = RetrievalRequest(
            query=case["query"],
            intent=case["intent"],
            jurisdiction=case.get("jurisdiction"),
            document_ids=case.get("document_ids", []),
            recommendation_ids=case.get("recommendation_ids", []),
            top_k=10,
        )
        result = retriever.retrieve(request)
        for name, ms in result.diagnostics.latency_ms.items():
            if name in latencies:
                latencies[name].append(ms)
        rows.append(
            {
                **case,
                "accepted": result.accepted,
                "ranked_documents": [item.unit.document_id for item in result.evidence],
                "ranked_ids": [item.unit.evidence_unit_id for item in result.evidence],
                "top_cosine": result.evidence[0].cosine_similarity if result.evidence else None,
                "reranker_used": result.diagnostics.reranker_used,
            }
        )
    return rows, {
        name: {"median_ms": median(values), "p95_ms": percentile(values, 95)}
        for name, values in latencies.items()
        if values
    }


class OldRetriever:
    def __init__(self, registry, embedder):
        self.registry, self.embedder = registry, embedder
        self.rows = []
        self.vectors = []
        with tempfile.TemporaryDirectory() as directory:
            for label in ("clinical", "anatomy"):
                index_bytes = subprocess.check_output(
                    ["git", "show", f"{BASELINE_SHA}:data/vector_store/{label}_faiss.index"], cwd=ROOT
                )
                metadata_bytes = subprocess.check_output(
                    ["git", "show", f"{BASELINE_SHA}:data/vector_store/{label}_metadata.json"], cwd=ROOT
                )
                path = Path(directory) / f"{label}.index"
                path.write_bytes(index_bytes)
                index = faiss.read_index(str(path))
                metadata = json.loads(metadata_bytes)
                self.rows += metadata
                self.vectors.extend(index.reconstruct(i) for i in range(index.ntotal))
        self.vectors = np.asarray(self.vectors, dtype="float32")

    def retrieve(self, request):
        selected = [
            i
            for i, row in enumerate(self.rows)
            if self.registry.eligible(row, request.policy)
            and (not request.document_ids or row["document_id"] in request.document_ids)
            and (not request.recommendation_ids or row.get("recommendation_id") in request.recommendation_ids)
        ]
        if not selected:
            return False, []
        rows = [self.rows[i] for i in selected]
        vectors = self.vectors[selected]
        q = np.asarray(self.embedder.encode([request.query]), dtype="float32")[0]
        distances = np.sum((vectors - q) ** 2, axis=1)
        dense = list(np.argsort(distances)[:20])
        threshold = 1.38 if request.policy is RetrievalPolicy.REFERENCE else 1.70
        if distances[dense[0]] > threshold:
            return False, []
        bm25 = BM25Okapi([simple_tokenize(row["text"]) for row in rows])
        scores = bm25.get_scores(simple_tokenize(request.query))
        lexical = [int(i) for i in np.argsort(-scores)[:20] if scores[i] > 0]
        fused = {}
        for ranking, weight in ((dense, 0.7), (lexical, 1.0)):
            for rank, i in enumerate(ranking, 1):
                fused[i] = fused.get(i, 0) + weight / (60 + rank)
        ranked = sorted(fused, key=lambda i: -fused[i])[: request.top_k]
        return bool(ranked), [rows[i]["document_id"] for i in ranked]


def main():
    registry = SourceRegistry()
    retriever = EvidenceRetriever(use_reranker=True)
    dev, _ = summarize(cases("development"), retriever)
    held, latency = summarize(cases("held_out"), retriever)
    fallback_rows, _ = summarize(
        cases("held_out"),
        EvidenceRetriever(
            registry=registry,
            embedder=retriever.embedder,
            use_reranker=False,
            units_and_vectors=(retriever.units, retriever.vectors),
        ),
    )
    old = OldRetriever(registry, retriever.embedder)
    baseline = []
    for case in cases("held_out"):
        request = RetrievalRequest(
            query=case["query"],
            intent=case["intent"],
            jurisdiction=case.get("jurisdiction"),
            document_ids=case.get("document_ids", []),
            recommendation_ids=case.get("recommendation_ids", []),
            top_k=10,
        )
        accepted, ranked = old.retrieve(request)
        baseline.append({**case, "accepted": accepted, "ranked_documents": ranked})
    report = {
        "schema_version": 1,
        "baseline_sha": BASELINE_SHA,
        "environment": {
            "os": platform.platform(),
            "processor": platform.processor(),
            "logical_cpus": os.cpu_count(),
            "python": sys.version.split()[0],
            "timing": "warm local CPU inference; held-out requests; no generation or verifier network time",
        },
        "calibration": {"split": "development", "cases": len(dev), "results": dev},
        "held_out": {
            "cases": len(held),
            "previous": metrics(baseline),
            "previous_results": baseline,
            "rrf_only": metrics(fallback_rows),
            "phase4": metrics(held),
            "results": held,
            "latency": latency,
        },
    }
    path = ROOT / "eval/retrieval_report.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "previous": metrics(baseline),
                "rrf_only": metrics(fallback_rows),
                "phase4": metrics(held),
                "latency": latency,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
