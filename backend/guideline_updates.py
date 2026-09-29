"""Human-governed recommendation releases; online acquisition never activates."""

import hashlib
import json
from dataclasses import replace
from difflib import unified_diff
from uuid import UUID

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import select, text

from backend.db import GuidelineActivation, GuidelineCandidate, session_factory, transaction, utcnow
from backend.durable_audit import append_event, canonical
from backend.evidence_models import EvidenceUnit, EvidenceUnitType
from backend.knowledge_models import IngestionStatus, KnowledgeDocumentVersion, Lifecycle
from backend.safe_http import AcquisitionError, SafeHTTP, origin
from backend.security import ActorContext
from backend.source_registry import SourceRegistry


class Recommendation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    recommendation_id: str | None = Field(default=None, pattern=r"^\d{1,2}\.\d{1,2}\.\d{1,3}$")
    section: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=25, max_length=2500)
    page: int = Field(ge=1, le=1000)


class CandidateInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    document_id: str = Field(pattern=r"^[a-z0-9-]{1,100}$")
    version_id: str = Field(pattern=r"^[a-z0-9.-]{1,180}$")
    source_url: str = Field(max_length=2000)
    updated_at: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    license_note: str = Field(min_length=10, max_length=1000)
    recommendations: list[Recommendation] = Field(min_length=1, max_length=300)

    @model_validator(mode="after")
    def valid_date(self):
        from datetime import date

        date.fromisoformat(self.updated_at)
        return self


def normalized(value: str) -> str:
    return " ".join(value.split())


def recommendation_diff(old: list[dict], new: list[dict]) -> dict:
    result: dict = {key: [] for key in ("added", "removed", "modified", "unchanged", "ambiguous")}

    def keyed(rows):
        groups: dict[str, list] = {}
        for row in rows:
            key = row.get("recommendation_id") or "section:" + row.get("section", "")
            groups.setdefault(key, []).append(row)
        return groups

    left, right = keyed(old), keyed(new)
    for key in sorted(left.keys() | right.keys()):
        before, after = left.get(key, []), right.get(key, [])
        if len(before) > 1 or len(after) > 1 or key == "section:":
            result["ambiguous"].append({"key": key, "old": before, "new": after})
        elif not before:
            result["added"].append(after[0])
        elif not after:
            result["removed"].append(before[0])
        elif normalized(before[0]["text"]) == normalized(after[0]["text"]):
            result["unchanged"].append({"key": key})
        elif key.startswith("section:"):
            # A section label alone does not establish recommendation identity.
            result["ambiguous"].append({"key": key, "old": before, "new": after})
        else:
            result["modified"].append(
                {
                    "key": key,
                    "old": before[0]["text"],
                    "new": after[0]["text"],
                    "text_diff": "\n".join(
                        unified_diff(
                            before[0]["text"].splitlines(),
                            after[0]["text"].splitlines(),
                            fromfile="old",
                            tofile="new",
                            lineterm="",
                        )
                    ),
                }
            )
    return result


def affected_rules(document_id: str, diff: dict) -> list[dict]:
    from backend.clinical_rules import default_registry

    changed = {r["key"] for r in diff["modified"]} | {r.get("recommendation_id") for r in diff["removed"]}
    return [
        {
            "rule_id": r.definition.rule_id,
            "rule_version": r.definition.rule_version,
            "evidence_id": r.definition.evidence_id,
            "recommendation_id": r.definition.recommendation_id,
        }
        for r in default_registry().current_rules()
        if r.definition.source_document_id == document_id
        and (r.definition.recommendation_id in changed or diff["ambiguous"])
    ]


def _audit(session, actor, request_id, event, candidate, note):
    append_event(
        session,
        event_type=event,
        resource_type="guideline_candidate",
        resource_id=str(candidate.candidate_id),
        actor_subject=actor.subject,
        actor_roles=sorted(actor.roles),
        request_id=request_id,
        payload={
            "document_id": candidate.document_id,
            "old_version": candidate.base_version_id,
            "new_version": candidate.version_id,
            "checksum": candidate.checksum,
            "note": note,
        },
    )


def require_editor(actor: ActorContext) -> None:
    if not actor.authenticated or not actor.roles.intersection({"guideline_editor", "clinical_admin"}):
        raise PermissionError("Authenticated guideline editor required")


def _base(session, registry, document_id):
    active = session.get(GuidelineActivation, document_id)
    if active:
        candidate = session.get(GuidelineCandidate, active.candidate_id)
        return candidate.version_id, candidate.payload["source"]["recommendations"]
    versions = [v for v in registry.versions.values() if v.document_id == document_id and v.status is Lifecycle.CURRENT]
    version = versions[0].version_id if versions else None
    units = json.loads((registry.root / "data/evidence_units.json").read_text(encoding="utf-8"))
    rows = [
        u
        for u in units
        if u["document_id"] == document_id and u["version_id"] == version and u["unit_type"] == "recommendation"
    ]
    return version, rows


def register_candidate(
    source: CandidateInput, actor: ActorContext, request_id: str, registry: SourceRegistry | None = None
) -> str:
    require_editor(actor)
    registry = registry or SourceRegistry()
    document = registry.documents.get(source.document_id)
    if document is None or document.source_type.value != "clinical_guideline" or not document.canonical_source_url:
        raise ValueError("Unregistered guideline")
    if origin(source.source_url) != origin(document.canonical_source_url):
        raise ValueError("Candidate origin differs from trusted source")
    if source.version_id in registry.versions:
        raise ValueError("A version identity cannot be reused")
    payload = source.model_dump(mode="json")
    checksum = hashlib.sha256(canonical(payload)).hexdigest()
    with transaction() as session:
        base_id, old = _base(session, registry, source.document_id)
        diff = recommendation_diff(old, payload["recommendations"])
        metadata_changes = []
        if not any(diff[key] for key in ("added", "removed", "modified", "ambiguous")):
            metadata_changes.append({"field": "version_id", "old": base_id, "new": source.version_id})
        diff.update(
            old_version=base_id,
            new_version=source.version_id,
            metadata_only_changes=metadata_changes,
            affected_rules=affected_rules(source.document_id, diff),
        )
        candidate = GuidelineCandidate(
            document_id=source.document_id,
            version_id=source.version_id,
            base_version_id=base_id,
            status="quarantined",
            checksum=checksum,
            payload={"source": payload},
            diff=diff,
        )
        session.add(candidate)
        session.flush()
        _audit(
            session,
            actor,
            request_id,
            "guideline_quarantined",
            candidate,
            "Manual structured source import; not activated",
        )
        return str(candidate.candidate_id)


def prepare(candidate_id: str, actor: ActorContext, request_id: str, embedder=None) -> None:
    require_editor(actor)
    with session_factory()() as session:
        candidate = session.get(GuidelineCandidate, UUID(candidate_id))
        if candidate is None or candidate.status != "quarantined":
            raise ValueError("Candidate is not quarantined")
        source = CandidateInput.model_validate(candidate.payload["source"])
        expected_hash = candidate.checksum
    if hashlib.sha256(canonical(source.model_dump(mode="json"))).hexdigest() != expected_hash:
        raise ValueError("Candidate checksum mismatch")
    if embedder is None:
        from sentence_transformers import SentenceTransformer
        from backend.index_provenance import EMBEDDING_MODEL
        from backend.model_versions import REVISIONS

        embedder = SentenceTransformer(EMBEDDING_MODEL, revision=REVISIONS[EMBEDDING_MODEL], trust_remote_code=False)
    vectors = np.asarray(
        embedder.encode([r.text for r in source.recommendations], normalize_embeddings=True), dtype="float32"
    )
    if (
        vectors.shape != (len(source.recommendations), 384)
        or not np.isfinite(vectors).all()
        or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-3)
    ):
        raise ValueError("Invalid candidate embeddings")
    with transaction() as session:
        candidate = session.scalar(
            select(GuidelineCandidate).where(GuidelineCandidate.candidate_id == UUID(candidate_id)).with_for_update()
        )
        if candidate is None or candidate.status != "quarantined" or candidate.checksum != expected_hash:
            raise ValueError("Candidate changed during preparation")
        candidate.payload = {
            "source": source.model_dump(mode="json"),
            "vectors": vectors.tolist(),
            "embedding_model": "all-MiniLM-L6-v2",
            "vector_sha256": hashlib.sha256(vectors.tobytes()).hexdigest(),
        }
        candidate.status = "review_required"
        _audit(
            session,
            actor,
            request_id,
            "guideline_diff_ready",
            candidate,
            "Bounded JSON parsed, diff and normalized vectors prepared",
        )


def decide(candidate_id: str, action: str, note: str, actor: ActorContext, request_id: str) -> dict:
    require_editor(actor)
    if not 5 <= len(note.strip()) <= 1000 or action not in {"approve", "reject", "activate"}:
        raise ValueError("Decision and meaningful review note required")
    with transaction() as session:
        candidate = session.scalar(
            select(GuidelineCandidate).where(GuidelineCandidate.candidate_id == UUID(candidate_id)).with_for_update()
        )
        if candidate is None:
            raise ValueError("Candidate not found")
        expected = {"approve": "review_required", "reject": "review_required", "activate": "approved"}[action]
        if candidate.status != expected:
            raise ValueError("Invalid candidate state transition")
        if action == "approve" and candidate.diff["ambiguous"]:
            raise ValueError("Ambiguous recommendations require a corrected new candidate")
        if action == "activate":
            lock = int.from_bytes(hashlib.sha256(candidate.document_id.encode()).digest()[:8], "big", signed=True)
            session.execute(text("SELECT pg_advisory_xact_lock(:key)"), {"key": lock})
            current, _ = _base(session, SourceRegistry(), candidate.document_id)
            if current != candidate.base_version_id:
                raise ValueError("Active guideline changed; rebase and review a new candidate")
            active = session.get(GuidelineActivation, candidate.document_id)
            if active:
                active.candidate_id, active.activated_at = candidate.candidate_id, utcnow()
            else:
                session.add(GuidelineActivation(document_id=candidate.document_id, candidate_id=candidate.candidate_id))
        candidate.status = {"approve": "approved", "reject": "rejected", "activate": "activated"}[action]
        _audit(session, actor, request_id, "guideline_" + candidate.status, candidate, note)
        return {"candidate_id": candidate_id, "status": candidate.status}


def active_overlay(registry: SourceRegistry) -> tuple[SourceRegistry, list[EvidenceUnit], np.ndarray]:
    """One database snapshot controls lifecycle and the complete unit/vector release."""
    import copy

    result = copy.copy(registry)
    result.versions = dict(registry.versions)
    with session_factory()() as session:
        rows = session.execute(
            select(GuidelineCandidate, GuidelineActivation.candidate_id.label("active_id"))
            .join(GuidelineActivation, GuidelineCandidate.document_id == GuidelineActivation.document_id)
            .where(GuidelineCandidate.status == "activated")
        ).all()
        units: list[EvidenceUnit] = []
        vectors: list[np.ndarray] = []
        replaced_documents = {candidate.document_id for candidate, _ in rows}
        for ident, version in list(result.versions.items()):
            if version.document_id in replaced_documents and version.status is Lifecycle.CURRENT:
                result.versions[ident] = replace(version, status=Lifecycle.SUPERSEDED)
        for candidate, active_id in rows:
            source = CandidateInput.model_validate(candidate.payload["source"])
            if hashlib.sha256(canonical(source.model_dump(mode="json"))).hexdigest() != candidate.checksum:
                raise ValueError("Activated source integrity mismatch")
            matrix = np.asarray(candidate.payload["vectors"], dtype="float32")
            if (
                candidate.payload.get("embedding_model") != "all-MiniLM-L6-v2"
                or matrix.shape != (len(source.recommendations), 384)
                or not np.isfinite(matrix).all()
                or not np.allclose(np.linalg.norm(matrix, axis=1), 1, atol=1e-3)
            ):
                raise ValueError("Activated embedding configuration mismatch")
            if hashlib.sha256(matrix.tobytes()).hexdigest() != candidate.payload["vector_sha256"]:
                raise ValueError("Activated vector integrity mismatch")
            lifecycle = Lifecycle.CURRENT if candidate.candidate_id == active_id else Lifecycle.SUPERSEDED
            result.versions[source.version_id] = KnowledgeDocumentVersion(
                version_id=source.version_id,
                document_id=source.document_id,
                version_label=source.updated_at,
                published_at=None,
                updated_at=source.updated_at,
                retrieved_at=candidate.created_at.date().isoformat(),
                valid_from=None,
                valid_until=None,
                status=lifecycle,
                supersedes_version_id=candidate.base_version_id,
                superseded_by_version_id=None,
                sha256=candidate.checksum,
                parser_version="structured-recommendations-v1",
                ingestion_status=IngestionStatus.INGESTED,
                source_path="postgresql:guideline_candidates",
                legacy_source=source.version_id,
            )
            doc = registry.documents[source.document_id]
            for i, rec in enumerate(source.recommendations):
                ident = (
                    f"{source.version_id}:rec:{rec.recommendation_id}"
                    if rec.recommendation_id
                    else f"{source.version_id}:section:{i}"
                )
                units.append(
                    EvidenceUnit(
                        evidence_unit_id=ident,
                        unit_type=EvidenceUnitType.RECOMMENDATION
                        if rec.recommendation_id
                        else EvidenceUnitType.CONTEXT_CHUNK,
                        recommendation_id=rec.recommendation_id,
                        source_chunk_ids=[f"{source.version_id}:section:{i}"],
                        document_id=source.document_id,
                        version_id=source.version_id,
                        text=rec.text,
                        section=rec.section,
                        page_start=rec.page,
                        page_end=rec.page,
                        source_type=doc.source_type,
                        jurisdiction=doc.jurisdiction,
                        lifecycle_status=lifecycle,
                        canonical_source_url=doc.canonical_source_url,
                        publisher=doc.publisher,
                        canonical_title=doc.canonical_title,
                        updated_at=source.updated_at,
                        extractor_version="structured-recommendations-v1",
                    )
                )
            vectors.extend(matrix)
    return result, units, np.asarray(vectors, dtype="float32").reshape(-1, 384)


def rule_needs_review(document_id: str) -> bool:
    # Conservative: any activated replacement suppresses dependent rules until
    # a versioned code/evidence review incorporates the new release explicitly.
    with session_factory()() as session:
        return session.get(GuidelineActivation, document_id) is not None


def acquire_registered(document_id: str, url: str) -> bytes:
    registry = SourceRegistry()
    doc = registry.documents.get(document_id)
    if doc is None or not doc.canonical_source_url or origin(url) != origin(doc.canonical_source_url):
        raise AcquisitionError("Unregistered guideline origin")
    # Deliberately support structured imports only. PDF/archive acquisition is
    # not exposed: no embedded programs, decompression or unbounded PDF parser.
    return SafeHTTP({origin(doc.canonical_source_url)}, max_bytes=1_000_000).request(url).body
