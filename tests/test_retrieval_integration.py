import os

import pytest

from backend.knowledge_models import RetrievalPolicy


@pytest.mark.skipif(os.environ.get("RUN_MODEL_INTEGRATION") != "1", reason="Requires locally cached embedding model")
def test_real_index_policy_and_api_smoke():
    from embeddings.retrieve import retrieve
    from api_server import health, sources

    current = retrieve("How should a diabetic foot ulcer be managed?", top_k=3)
    assert current and all(
        c["source_type"] == "clinical_guideline" and c["lifecycle_status"] == "current" for c in current
    )
    assert not any(c["document_id"] == "nice-ng28" for c in current)
    assert retrieve("What is the SINBAD classification?")[0]["document_id"] == "nice-ng19"
    assert retrieve("What is the capital of France?") == []
    assert current[0]["chunk_id"] and current[0]["publisher"] == "NICE"
    assert current[0]["page_start"] >= 1 and current[0]["source_url"]
    old = retrieve("World malaria report 2019", policy=RetrievalPolicy.HISTORICAL_ONLY)
    assert old and all(c["lifecycle_status"] in ("historical", "superseded") for c in old)
    reference = retrieve("Explain the anatomy of the heart", policy=RetrievalPolicy.REFERENCE)
    assert reference and all(c["source_type"] in ("textbook", "patient_education") for c in reference)
    assert health() == {"status": "ok"}
    assert len(sources()["sources"]) == 3
