"""Optional local second-stage cross-encoder, with explicit fallback."""

from typing import TYPE_CHECKING, Protocol
from threading import Lock

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

from backend.evidence_models import RankedEvidence

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class EvidenceReranker(Protocol):
    def rerank(self, query: str, candidates: list[RankedEvidence], top_k: int) -> tuple[list[RankedEvidence], bool]: ...


class RRFFallbackReranker:
    def rerank(self, query: str, candidates: list[RankedEvidence], top_k: int) -> tuple[list[RankedEvidence], bool]:
        if top_k <= 0:
            return [], False
        return sorted(candidates, key=lambda c: (-c.rrf_score, c.unit.evidence_unit_id))[:top_k], False


class CrossEncoderReranker:
    def __init__(self, model_name: str | None = None):
        from backend.settings import get_settings

        self.model_name = model_name or get_settings().reranker_model
        self._model: CrossEncoder | None = None
        self._lock = Lock()

    def rerank(self, query: str, candidates: list[RankedEvidence], top_k: int) -> tuple[list[RankedEvidence], bool]:
        if not candidates or top_k <= 0:
            return [], False
        try:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import CrossEncoder

                    self._model = CrossEncoder(self.model_name, max_length=256)
                model = self._model
                pairs = [(query, c.unit.text[:1200]) for c in candidates]
                try:
                    scores = model.predict(pairs, batch_size=32)
                except TypeError:
                    scores = model.predict(pairs)
        except Exception:
            return RRFFallbackReranker().rerank(query, candidates, top_k)
        ranked = [
            candidate.model_copy(update={"reranker_score": float(score)})
            for candidate, score in zip(candidates, scores, strict=True)
        ]
        return sorted(
            ranked,
            key=lambda c: (
                -(c.reranker_score if c.reranker_score is not None else float("-inf")),
                c.unit.evidence_unit_id,
            ),
        )[:top_k], True
