"""Independent claim verification with deterministic provenance checks first."""

import re
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

import numpy as np

from backend.evidence_models import AnswerClaim, EvidenceUnit, RetrievalResult, SupportStatus
from backend.knowledge_models import Lifecycle
from backend.source_registry import RegistryError, SourceRegistry

NLI_MODEL = "cross-encoder/nli-MiniLM2-L6-H768"
VERIFICATION_PASSAGE_CHARS = 850


class ClaimSupportVerifier(Protocol):
    def verify(self, claim: AnswerClaim, evidence: list[EvidenceUnit]) -> SupportStatus: ...


def supporting_passage(text: str, claim: str, max_chars: int = VERIFICATION_PASSAGE_CHARS) -> str:
    """Select a bounded passage for both verification and evidence display."""
    if len(text) <= max_chars:
        return text
    tokens = set(re.findall(r"[a-z0-9]+", claim.lower()))
    windows = [text[i : i + max_chars] for i in range(0, len(text), max_chars - 150)]
    return max(windows, key=lambda part: len(tokens & set(re.findall(r"[a-z0-9]+", part.lower()))))


class NLIClaimVerifier:
    def __init__(self, model_name: str = NLI_MODEL):
        self.model_name = model_name
        self._model: CrossEncoder | None = None

    def verify(self, claim: AnswerClaim, evidence: list[EvidenceUnit]) -> SupportStatus:
        if not evidence:
            return SupportStatus.UNSUPPORTED
        try:
            if self._model is None:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(self.model_name, max_length=512)
            pairs = [(supporting_passage(unit.text, claim.text), claim.text) for unit in evidence]
            model = self._model
            scores = np.asarray(model.predict(pairs, apply_softmax=True))
            labels = {str(name).lower(): int(index) for index, name in model.model.config.id2label.items()}
            entailment = next(index for name, index in labels.items() if "entail" in name)
            contradiction = next(index for name, index in labels.items() if "contrad" in name)
        except Exception:
            return SupportStatus.UNCERTAIN
        # Every cited unit must independently support the claim. This
        # intentionally rejects claims assembled from partial passages until
        # a separately validated multi-premise verifier is available.
        if np.all((scores[:, entailment] >= 0.85) & (scores[:, entailment] > scores[:, contradiction])):
            return SupportStatus.SUPPORTED
        return SupportStatus.UNSUPPORTED if np.any(scores[:, contradiction] >= 0.70) else SupportStatus.UNCERTAIN

    def contradicts(self, first: str, second: str) -> bool:
        try:
            if self._model is None:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(self.model_name, max_length=512)
            model = self._model
            scores = np.asarray(model.predict([(first, second), (second, first)], apply_softmax=True))
            labels = {str(name).lower(): int(index) for index, name in model.model.config.id2label.items()}
            contradiction = next(index for name, index in labels.items() if "contrad" in name)
            return bool(np.all(scores[:, contradiction] >= 0.85))
        except Exception:
            return False


def verify_claim(
    claim: AnswerClaim,
    retrieval: RetrievalResult,
    semantic: ClaimSupportVerifier,
    registry: SourceRegistry | None = None,
) -> AnswerClaim:
    available = {ranked.unit.evidence_unit_id: ranked.unit for ranked in retrieval.evidence}
    if not claim.evidence_ids or len(set(claim.evidence_ids)) != len(claim.evidence_ids):
        return claim.model_copy(update={"support_status": SupportStatus.UNSUPPORTED})
    units = []
    for ident in claim.evidence_ids:
        unit = available.get(ident)
        if (
            unit is None
            or not unit.text
            or not unit.document_id
            or not unit.version_id
            or unit.page_start is None
            or unit.page_end is None
        ):
            return claim.model_copy(update={"support_status": SupportStatus.UNSUPPORTED})
        if registry is not None:
            try:
                document, version = registry.resolve(unit.model_dump())
                if (
                    document.document_id != unit.document_id
                    or version.version_id != unit.version_id
                    or version.status != unit.lifecycle_status
                    or not registry.eligible(unit.model_dump(), retrieval.request.policy)
                ):
                    return claim.model_copy(update={"support_status": SupportStatus.UNSUPPORTED})
            except RegistryError:
                return claim.model_copy(update={"support_status": SupportStatus.UNSUPPORTED})
        try:
            # The retriever has already checked registry identity; recheck the
            # policy boundary using immutable unit metadata before semantics.
            if retrieval.request.policy.value == "current_clinical" and unit.lifecycle_status is not Lifecycle.CURRENT:
                return claim.model_copy(update={"support_status": SupportStatus.UNSUPPORTED})
            if retrieval.request.policy.value == "historical_only" and unit.lifecycle_status not in (
                Lifecycle.HISTORICAL,
                Lifecycle.SUPERSEDED,
            ):
                return claim.model_copy(update={"support_status": SupportStatus.UNSUPPORTED})
        except ValueError:
            return claim.model_copy(update={"support_status": SupportStatus.UNSUPPORTED})
        units.append(unit)
    passages = {unit.evidence_unit_id: supporting_passage(unit.text, claim.text) for unit in units}
    return claim.model_copy(update={"support_status": semantic.verify(claim, units), "verification_passages": passages})
