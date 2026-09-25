"""Lifecycle-filtered cosine/BM25 retrieval and bounded rank fusion."""

import json
import re
from pathlib import Path
from time import perf_counter
from typing import Protocol

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from backend.evidence_models import (
    EvidenceCandidate,
    EvidenceUnit,
    RankedEvidence,
    RetrievalDiagnostics,
    RetrievalRequest,
    RetrievalResult,
)
from backend.index_provenance import validate_chunks_manifest, validate_manifest
from backend.knowledge_models import RetrievalPolicy
from backend.source_registry import SourceRegistry, StaleArtifact
from embeddings.reranker import CrossEncoderReranker, EvidenceReranker, RRFFallbackReranker

ROOT = Path(__file__).resolve().parent.parent
VECTOR_DIR = ROOT / "data/vector_store"
CANDIDATE_POOL_SIZE = 30
RRF_K = 60
RRF_DENSE_WEIGHT = 0.7
RRF_BM25_WEIGHT = 1.0
# Selected by the development split in eval/calibrate.py; held-out results
# are reported separately. Cosine is a similarity, not a probability.
MIN_COSINE = 0.40


class Embedder(Protocol):
    def encode(self, sentences: list[str], *, normalize_embeddings: bool) -> object: ...


class DenseCandidateRetriever(Protocol):
    def retrieve(
        self, units: list[EvidenceUnit], vectors: np.ndarray, query_vector: np.ndarray, limit: int
    ) -> list[EvidenceCandidate]: ...


class BM25CandidateRetriever(Protocol):
    def retrieve(self, units: list[EvidenceUnit], query: str, limit: int) -> list[EvidenceCandidate]: ...


class CosineDenseRetriever:
    def retrieve(
        self, units: list[EvidenceUnit], vectors: np.ndarray, query_vector: np.ndarray, limit: int
    ) -> list[EvidenceCandidate]:
        similarities = vectors @ query_vector[0]
        order = np.argsort(-similarities, kind="stable")[:limit]
        return [
            EvidenceCandidate(
                evidence_unit_id=units[int(i)].evidence_unit_id,
                retriever="dense",
                rank=rank,
                raw_score=float(similarities[i]),
            )
            for rank, i in enumerate(order, 1)
        ]


class BM25LexicalRetriever:
    def retrieve(self, units: list[EvidenceUnit], query: str, limit: int) -> list[EvidenceCandidate]:
        bm25_index = BM25Okapi([simple_tokenize(unit.text) for unit in units])
        scores = bm25_index.get_scores(simple_tokenize(query))
        order = np.argsort(-scores, kind="stable")[:limit]
        return [
            EvidenceCandidate(
                evidence_unit_id=units[int(i)].evidence_unit_id,
                retriever="bm25",
                rank=rank,
                raw_score=float(scores[i]),
            )
            for rank, i in enumerate(order, 1)
            if scores[i] > 0
        ]


def simple_tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def rrf_fuse(dense: list[EvidenceCandidate], bm25: list[EvidenceCandidate]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for candidates, weight in ((dense, RRF_DENSE_WEIGHT), (bm25, RRF_BM25_WEIGHT)):
        for candidate in candidates:
            ident = candidate.evidence_unit_id
            scores[ident] = scores.get(ident, 0.0) + weight / (RRF_K + candidate.rank)
    return scores


class EvidenceRetriever:
    def __init__(
        self,
        *,
        registry: SourceRegistry | None = None,
        embedder: Embedder | None = None,
        reranker: EvidenceReranker | None = None,
        dense_retriever: DenseCandidateRetriever | None = None,
        bm25_retriever: BM25CandidateRetriever | None = None,
        use_reranker: bool = True,
        units_and_vectors: tuple[list[EvidenceUnit], np.ndarray] | None = None,
    ):
        self.registry = registry or SourceRegistry()
        self.embedder = embedder
        self.reranker = reranker or (CrossEncoderReranker() if use_reranker else RRFFallbackReranker())
        self.dense_retriever = dense_retriever or CosineDenseRetriever()
        self.bm25_retriever = bm25_retriever or BM25LexicalRetriever()
        if units_and_vectors is not None:
            self.units, self.vectors = units_and_vectors
        else:
            validate_chunks_manifest(ROOT / "data/chunks.json", self.registry)
            units: list[EvidenceUnit] = []
            matrices = []
            for label in ("clinical", "anatomy"):
                index_path = VECTOR_DIR / f"{label}_faiss.index"
                metadata_path = VECTOR_DIR / f"{label}_metadata.json"
                validate_manifest(index_path, metadata_path, self.registry)
                index = faiss.read_index(str(index_path))
                if not isinstance(index, faiss.IndexFlatIP):
                    raise StaleArtifact("Expected cosine inner-product index")
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                if index.ntotal != len(metadata):
                    raise StaleArtifact("Vector count differs from evidence metadata")
                units.extend(EvidenceUnit.model_validate(row) for row in metadata)
                matrices.append(np.stack([index.reconstruct(i) for i in range(index.ntotal)]))
            self.units = units
            self.vectors = np.concatenate(matrices).astype("float32")
        if len(self.units) != len(self.vectors):
            raise ValueError("Evidence/vector count mismatch")
        if len({unit.evidence_unit_id for unit in self.units}) != len(self.units):
            raise ValueError("Duplicate evidence unit ID")
        if not np.allclose(np.linalg.norm(self.vectors, axis=1), 1.0, atol=1e-3):
            raise StaleArtifact("Indexed evidence vectors are not normalized")

    def _encode(self, query: str) -> np.ndarray:
        if self.embedder is None:
            from sentence_transformers import SentenceTransformer

            self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        embedder = self.embedder
        assert embedder is not None
        vector = np.asarray(embedder.encode([query], normalize_embeddings=True), dtype="float32")
        norm = np.linalg.norm(vector[0])
        if norm <= 0:
            raise ValueError("Empty query embedding")
        return vector / norm

    def _eligible(self, unit: EvidenceUnit, request: RetrievalRequest) -> bool:
        if not self.registry.eligible(unit.model_dump(), request.policy):
            return False
        return not (
            (request.jurisdiction and unit.jurisdiction != request.jurisdiction)
            or (request.source_types and unit.source_type not in request.source_types)
            or (request.document_ids and unit.document_id not in request.document_ids)
            or (request.version_ids and unit.version_id not in request.version_ids)
            or (request.recommendation_ids and unit.recommendation_id not in request.recommendation_ids)
            or (request.unit_types and unit.unit_type not in request.unit_types)
        )

    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        began = perf_counter()
        selected = [i for i, unit in enumerate(self.units) if self._eligible(unit, request)]
        diagnostics = RetrievalDiagnostics()
        if not selected:
            diagnostics.acceptance_reason = "no_eligible_evidence"
            return RetrievalResult(request=request, accepted=False, evidence=[], diagnostics=diagnostics)
        units = [self.units[i] for i in selected]
        vectors = self.vectors[selected]
        started = perf_counter()
        query_vector = self._encode(request.query)
        dense = self.dense_retriever.retrieve(units, vectors, query_vector, CANDIDATE_POOL_SIZE)
        diagnostics.latency_ms["dense"] = (perf_counter() - started) * 1000
        started = perf_counter()
        bm25 = self.bm25_retriever.retrieve(units, request.query, CANDIDATE_POOL_SIZE)
        diagnostics.latency_ms["bm25"] = (perf_counter() - started) * 1000
        diagnostics.dense_count, diagnostics.bm25_count = len(dense), len(bm25)
        diagnostics.top_cosine_similarity = dense[0].raw_score if dense else None
        fused = rrf_fuse(dense, bm25)
        by_id = {unit.evidence_unit_id: unit for unit in units}
        dense_map = {c.evidence_unit_id: c for c in dense}
        bm25_map = {c.evidence_unit_id: c for c in bm25}
        first_stage = [
            RankedEvidence(
                unit=by_id[ident],
                dense_rank=dense_map[ident].rank if ident in dense_map else None,
                bm25_rank=bm25_map[ident].rank if ident in bm25_map else None,
                cosine_similarity=dense_map[ident].raw_score if ident in dense_map else None,
                bm25_score=bm25_map[ident].raw_score if ident in bm25_map else None,
                rrf_score=score,
            )
            for ident, score in sorted(fused.items(), key=lambda pair: (-pair[1], pair[0]))
        ]
        diagnostics.candidate_count = len(first_stage)
        # BM25 cannot rescue a semantically unrelated query through a few shared words.
        if not dense or dense[0].raw_score < MIN_COSINE:
            diagnostics.acceptance_reason = "cosine_below_calibrated_floor"
            diagnostics.latency_ms["full_retrieval"] = (perf_counter() - began) * 1000
            return RetrievalResult(request=request, accepted=False, evidence=[], diagnostics=diagnostics)
        started = perf_counter()
        ranked, diagnostics.reranker_used = self.reranker.rerank(
            request.query, first_stage[:CANDIDATE_POOL_SIZE], request.top_k
        )
        diagnostics.latency_ms["reranking"] = (perf_counter() - started) * 1000
        diagnostics.latency_ms["full_retrieval"] = (perf_counter() - began) * 1000
        diagnostics.acceptance_reason = "eligible_cosine_match" if ranked else "no_ranked_evidence"
        return RetrievalResult(request=request, accepted=bool(ranked), evidence=ranked, diagnostics=diagnostics)


_default_retriever: EvidenceRetriever | None = None


def get_default_retriever() -> EvidenceRetriever:
    global _default_retriever
    if _default_retriever is None:
        _default_retriever = EvidenceRetriever()
    return _default_retriever


def retrieve(query: str, top_k: int = 5, policy: RetrievalPolicy = RetrievalPolicy.CURRENT_CLINICAL) -> list[dict]:
    """Compatibility interface; canonical callers use RetrievalResult."""
    result = get_default_retriever().retrieve(RetrievalRequest(query=query, lifecycle_policy=policy, top_k=top_k))
    rows = [item.unit.model_dump(mode="json") for item in result.evidence] if result.accepted else []
    for row in rows:
        row["source_url"] = row["canonical_source_url"]
        row["chunk_id"] = row["chunk_id"] or row["evidence_unit_id"]
    return rows
