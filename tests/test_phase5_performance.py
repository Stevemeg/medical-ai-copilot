"""Bounded lexical index reuse and safe invalidation."""

import json
from pathlib import Path

import numpy as np
import pytest

from backend.evidence_models import EvidenceUnit
from backend.evidence_models import RetrievalRequest
from backend.source_registry import StaleArtifact
from embeddings.retrieve import BM25LexicalRetriever, EvidenceRetriever


def test_bm25_partition_reuse_and_text_invalidation():
    rows = json.loads(Path("data/vector_store/clinical_metadata.json").read_text(encoding="utf-8"))
    units = [EvidenceUnit.model_validate(row) for row in rows[:3]]
    lexical = BM25LexicalRetriever(max_partitions=2)
    lexical.retrieve(units, "diabetes care", 2)
    original = next(iter(lexical._cache.values()))
    lexical.retrieve(units, "hypertension", 2)
    assert len(lexical._cache) == 1 and next(iter(lexical._cache.values())) is original
    changed = [units[0].model_copy(update={"text": units[0].text + " changed"}), *units[1:]]
    lexical.retrieve(changed, "diabetes", 2)
    assert len(lexical._cache) == 2
    lexical.retrieve(units[:1], "diabetes", 2)
    assert len(lexical._cache) == 2
    assert original not in lexical._cache.values()


def test_retriever_rejects_changed_manifest_signature(monkeypatch):
    row = json.loads(Path("data/vector_store/clinical_metadata.json").read_text(encoding="utf-8"))[0]
    unit = EvidenceUnit.model_validate(row)
    retriever = EvidenceRetriever(use_reranker=False, units_and_vectors=([unit], np.asarray([[1.0]], dtype="float32")))
    retriever._artifact_signature = ("original",)
    monkeypatch.setattr(retriever, "_current_artifact_signature", lambda: ("changed",))
    with pytest.raises(StaleArtifact):
        retriever.retrieve(RetrievalRequest(query="hypertension"))
