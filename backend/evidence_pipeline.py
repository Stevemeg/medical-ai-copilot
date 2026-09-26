"""Single evidence-answer service for API and UIs."""

from time import perf_counter
from typing import Protocol, cast
from prometheus_client import Counter, Histogram
from opentelemetry import trace

from backend.conflicts import SemanticContradiction, detect_conflicts
from backend.evidence_models import (
    AnswerClaim,
    AnswerStatus,
    EvidenceAnswer,
    RetrievalRequest,
    RetrievalResult,
    SupportStatus,
)
from backend.generator import AnswerGenerator, GroqAnswerGenerator, SYSTEM_PROMPT, parse_draft, render_evidence_prompt
from backend.grounding import ClaimSupportVerifier, NLIClaimVerifier, supporting_passage, verify_claim

ABSTENTION = "The indexed evidence does not support a reliable answer."
EVIDENCE_ANSWERS = Counter("medical_evidence_answers_total", "Evidence answer status", ["status"])
VERIFICATION_LATENCY = Histogram("medical_verification_latency_seconds", "Claim verification latency")
TRACER = trace.get_tracer(__name__)


def log_interaction(*_args):
    """Legacy test seam; audit is owned by the authenticated API/UI boundary."""


class Retriever(Protocol):
    def retrieve(self, request: RetrievalRequest) -> RetrievalResult: ...


class EvidenceAnswerService:
    def __init__(
        self,
        retriever: Retriever,
        generator: AnswerGenerator | None = None,
        verifier: ClaimSupportVerifier | None = None,
    ):
        self.retriever = retriever
        self.generator = generator or GroqAnswerGenerator()
        self.verifier = verifier or NLIClaimVerifier()

    def _record(self, request: RetrievalRequest, answer: EvidenceAnswer) -> EvidenceAnswer:
        # API and UI boundaries attribute and persist minimized audit events.
        EVIDENCE_ANSWERS.labels(answer.status.value).inc()
        return answer

    def answer(self, request: RetrievalRequest) -> EvidenceAnswer:
        try:
            retrieval = self.retriever.retrieve(request)
        except Exception:
            return self._record(
                request,
                EvidenceAnswer(
                    status=AnswerStatus.ABSTAINED, answer_text=ABSTENTION, failure_reason="retrieval_failure"
                ),
            )
        if not retrieval.accepted:
            return self._record(
                request,
                EvidenceAnswer(
                    status=AnswerStatus.ABSTAINED,
                    answer_text=ABSTENTION,
                    retrieval=retrieval.diagnostics,
                    failure_reason="retrieval_abstention",
                ),
            )
        units = [item.unit for item in retrieval.evidence]
        semantic_conflict = (
            cast(SemanticContradiction, self.verifier) if hasattr(self.verifier, "contradicts") else None
        )
        conflicts = detect_conflicts(units, semantic_conflict)
        if conflicts:
            ids = {ident for item in conflicts for ident in item.evidence_ids}
            return self._record(
                request,
                EvidenceAnswer(
                    status=AnswerStatus.CONFLICT,
                    answer_text="Evidence differs across the cited sources. Review each source.",
                    conflicts=conflicts,
                    evidence=[unit for unit in units if unit.evidence_unit_id in ids],
                    retrieval=retrieval.diagnostics,
                    failure_reason="conflict",
                ),
            )
        draft = None
        for attempt in range(2):
            try:
                prompt = render_evidence_prompt(
                    request.query, units, "Return valid JSON conforming to the schema." if attempt else None
                )
                draft = parse_draft(self.generator.generate(SYSTEM_PROMPT, prompt))
                break
            except ValueError:
                if attempt == 1:
                    return self._record(
                        request,
                        EvidenceAnswer(
                            status=AnswerStatus.ABSTAINED,
                            answer_text=ABSTENTION,
                            retrieval=retrieval.diagnostics,
                            failure_reason="generation_failure",
                        ),
                    )
            except Exception:
                return self._record(
                    request,
                    EvidenceAnswer(
                        status=AnswerStatus.ABSTAINED,
                        answer_text=ABSTENTION,
                        retrieval=retrieval.diagnostics,
                        failure_reason="provider_failure",
                    ),
                )
        if draft is None or draft.status == "abstained":
            return self._record(
                request,
                EvidenceAnswer(
                    status=AnswerStatus.ABSTAINED,
                    answer_text=ABSTENTION,
                    retrieval=retrieval.diagnostics,
                    failure_reason="generation_abstention",
                ),
            )
        started = perf_counter()
        claims: list[AnswerClaim] = []
        with TRACER.start_as_current_span("evidence.verification") as span:
            span.set_attribute("claim_count", len(draft.claims))
            for raw in draft.claims:
                claim = AnswerClaim(claim_id=raw.claim_id, text=raw.text, evidence_ids=raw.evidence_ids)
                claims.append(verify_claim(claim, retrieval, self.verifier, getattr(self.retriever, "registry", None)))
        retrieval.diagnostics.latency_ms["verification"] = (perf_counter() - started) * 1000
        VERIFICATION_LATENCY.observe((perf_counter() - started))
        supported = [claim for claim in claims if claim.support_status is SupportStatus.SUPPORTED]
        rejected = [claim for claim in claims if claim.support_status is not SupportStatus.SUPPORTED]
        if not supported:
            return self._record(
                request,
                EvidenceAnswer(
                    status=AnswerStatus.ABSTAINED,
                    answer_text=ABSTENTION,
                    rejected_claims=rejected,
                    retrieval=retrieval.diagnostics,
                    failure_reason="verification_failure",
                ),
            )
        cited = {ident for claim in supported for ident in claim.evidence_ids}
        used = [unit for unit in units if unit.evidence_unit_id in cited]
        # The model's free-form answer is never rendered. Only verified claim
        # texts survive; jurisdiction labels prevent silent blending.
        by_id = {unit.evidence_unit_id: unit for unit in used}
        sentences = []
        for claim in supported:
            jurisdictions = sorted({by_id[ident].jurisdiction for ident in claim.evidence_ids})
            prefix = f"[{', '.join(jurisdictions)}] " if jurisdictions else ""
            sentences.append(prefix + claim.text.strip())
        return self._record(
            request,
            EvidenceAnswer(
                status=AnswerStatus.GROUNDED,
                answer_text="\n".join(sentences),
                claims=supported,
                rejected_claims=rejected,
                evidence=used,
                retrieval=retrieval.diagnostics,
            ),
        )


def public_answer(answer: EvidenceAnswer) -> dict:
    payload = answer.model_dump(mode="json")
    payload["answer"] = payload["answer_text"]
    for unit in payload["evidence"]:
        claim_passage = next(
            (
                claim.verification_passages[unit["evidence_unit_id"]]
                for claim in answer.claims
                if unit["evidence_unit_id"] in claim.verification_passages
            ),
            None,
        )
        unit["supporting_excerpt"] = claim_passage or supporting_passage(unit["text"], "")
        unit.pop("text", None)
    return payload
