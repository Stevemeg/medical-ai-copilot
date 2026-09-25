"""Compare lexical score gaps for exact-term and unrelated questions."""

from backend.knowledge_models import RetrievalPolicy
from embeddings.retrieve import BM25LexicalRetriever, EvidenceRetriever

IRRELEVANT_QUESTIONS = [
    "How do I repair a motorcycle engine?",
    "How long should I bake a chocolate cake?",
]
EXACT_TERM_QUESTIONS = [
    "ACEi hypertension",
    "QRISK3 cardiovascular prevention",
]


def main() -> None:
    retriever = EvidenceRetriever(use_reranker=False)
    units = [
        unit
        for unit in retriever.units
        if retriever.registry.eligible(unit.model_dump(), RetrievalPolicy.CURRENT_CLINICAL)
    ]
    lexical = BM25LexicalRetriever()
    for label, questions in (("unrelated", IRRELEVANT_QUESTIONS), ("exact term", EXACT_TERM_QUESTIONS)):
        print(label)
        for query in questions:
            rows = lexical.retrieve(units, query, 2)
            scores = [row.raw_score for row in rows]
            gap = scores[0] - scores[1] if len(scores) == 2 else None
            print(f"{query}: top_scores={scores} gap={gap}")


if __name__ == "__main__":
    main()
