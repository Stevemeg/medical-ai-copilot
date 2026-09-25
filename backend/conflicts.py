"""Conservative detection of materially incompatible evidence statements."""

import re
from typing import Protocol

from backend.evidence_models import EvidenceConflict, EvidenceUnit

STOP = {"the", "and", "for", "with", "every", "routine", "perform", "do", "not", "a", "an", "to", "of"}


class SemanticContradiction(Protocol):
    def contradicts(self, first: str, second: str) -> bool: ...


def _topic_tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower())) - STOP


def _opposed(a: str, b: str) -> bool:
    a, b = a.lower(), b.lower()
    annual_a = bool(re.search(r"\b(annual|annually|every 12 months)\b", a))
    annual_b = bool(re.search(r"\b(annual|annually|every 12 months)\b", b))
    neg_a = bool(re.search(r"\b(do not|never|avoid|should not)\b", a))
    neg_b = bool(re.search(r"\b(do not|never|avoid|should not)\b", b))
    return annual_a and annual_b and neg_a != neg_b


def detect_conflicts(
    units: list[EvidenceUnit], semantic: SemanticContradiction | None = None
) -> list[EvidenceConflict]:
    conflicts: list[EvidenceConflict] = []
    for i, first in enumerate(units):
        for second in units[i + 1 :]:
            if first.document_id == second.document_id and first.version_id == second.version_id:
                continue
            left, right = _topic_tokens(first.text), _topic_tokens(second.text)
            if not left or not right or len(left & right) / min(len(left), len(right)) < 0.25:
                continue
            annual_opposition = _opposed(first.text, second.text)
            if not (
                annual_opposition
                or (semantic is not None and semantic.contradicts(first.text[:1000], second.text[:1000]))
            ):
                continue
            conflicts.append(
                EvidenceConflict(
                    conflict_id=f"conflict-{len(conflicts) + 1}",
                    topic=" ".join(sorted(left & right)[:6]),
                    evidence_ids=[first.evidence_unit_id, second.evidence_unit_id],
                    jurisdictions=sorted({first.jurisdiction, second.jurisdiction}),
                    description=(
                        "One source recommends annual testing while another advises against it."
                        if annual_opposition
                        else "Relevant sources contain materially conflicting statements; review both."
                    ),
                )
            )
    return conflicts
