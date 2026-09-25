"""Inspect reference-mode retrieval for an insulin-resistance explanation."""

from backend.evidence_models import QueryIntent, RetrievalRequest
from embeddings.retrieve import EvidenceRetriever

QUERY = "Explain insulin resistance"


def main() -> None:
    retriever = EvidenceRetriever(use_reranker=False)
    result = retriever.retrieve(RetrievalRequest(query=QUERY, intent=QueryIntent.REFERENCE_EXPLANATION))
    print(f"accepted={result.accepted} reason={result.diagnostics.acceptance_reason}")
    print(f"top_cosine={result.diagnostics.top_cosine_similarity}")
    for item in result.evidence:
        print(
            f"{item.unit.evidence_unit_id}: dense={item.dense_rank} bm25={item.bm25_rank} "
            f"cosine={item.cosine_similarity} rrf={item.rrf_score:.4f}"
        )


if __name__ == "__main__":
    main()
