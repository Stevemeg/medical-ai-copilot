"""Legacy Ask Evidence adapter over the canonical verified evidence service."""

from threading import Lock

from backend.evidence_models import QueryIntent, RetrievalRequest
from backend.evidence_pipeline import EvidenceAnswerService, public_answer
from backend.knowledge_models import RetrievalPolicy

_service: EvidenceAnswerService | None = None
_service_lock = Lock()


def get_service() -> EvidenceAnswerService:
    global _service
    if _service is None:
        with _service_lock:
            if _service is None:
                from embeddings.retrieve import get_default_retriever

                _service = EvidenceAnswerService(get_default_retriever())
    return _service


def answer_question(question: str, top_k: int = 5, policy: RetrievalPolicy = RetrievalPolicy.CURRENT_CLINICAL) -> dict:
    intent = (
        QueryIntent.REFERENCE_EXPLANATION
        if policy is RetrievalPolicy.REFERENCE
        else QueryIntent.HISTORICAL
        if policy in (RetrievalPolicy.HISTORICAL_ALLOWED, RetrievalPolicy.HISTORICAL_ONLY)
        else QueryIntent.CLINICAL_GUIDANCE
    )
    answer = get_service().answer(RetrievalRequest(query=question, top_k=top_k, intent=intent, lifecycle_policy=policy))
    public = public_answer(answer)
    cited = {ident for claim in answer.claims for ident in claim.evidence_ids}
    sources = []
    citations = []
    for unit in answer.evidence:
        if unit.evidence_unit_id not in cited:
            continue
        page = (
            (
                f", page {unit.page_start}"
                if unit.page_start == unit.page_end
                else f", pages {unit.page_start}-{unit.page_end}"
            )
            if unit.page_start
            else ""
        )
        sources.append(f"{unit.canonical_title}{page}")
        citations.append(
            {
                "evidence_unit_id": unit.evidence_unit_id,
                "chunk_id": unit.chunk_id,
                "document_id": unit.document_id,
                "version_id": unit.version_id,
                "canonical_title": unit.canonical_title,
                "publisher": unit.publisher,
                "source_type": unit.source_type.value,
                "jurisdiction": unit.jurisdiction,
                "lifecycle_status": unit.lifecycle_status.value,
                "published_at": unit.published_at,
                "updated_at": unit.updated_at,
                "source_url": unit.canonical_source_url,
                "page_start": unit.page_start,
                "page_end": unit.page_end,
                "section": unit.section,
                "recommendation_id": unit.recommendation_id,
            }
        )
    return {
        "answer": answer.answer_text,
        "sources": sources,
        "citations": citations,
        "status": answer.status.value,
        "claims": public["claims"],
        "conflicts": public["conflicts"],
        "evidence": public["evidence"],
        "retrieval": public["retrieval"],
    }
