"""
Diagnostic script -- NOT part of the app. Traces what happens for
"What are the risk factors for hypertension?", since eval_generation_quality.py
showed this incorrectly routing to the anatomy index instead of the NICE
hypertension guideline in the clinical index.

Usage:
    python embeddings/debug_hypertension.py
"""

import numpy as np

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import retrieve

QUERY = "What are the risk factors for hypertension?"

print("=" * 70)
print("STEP 1: What does BM25 think, on EACH index?")
print("=" * 70)
clinical_bm25_scores = retrieve.clinical_bm25.get_scores(retrieve.simple_tokenize(QUERY))
clinical_bm25_top = np.argsort(clinical_bm25_scores)[::-1][:5]
print("Clinical BM25 top 5:")
for rank, idx in enumerate(clinical_bm25_top, start=1):
    source = retrieve.clinical_metadata[idx].get("source", "?")
    print(f"  rank {rank}: chunk_index={idx} score={clinical_bm25_scores[idx]:.3f}  source={source}")

anatomy_bm25_scores = retrieve.anatomy_bm25.get_scores(retrieve.simple_tokenize(QUERY))
anatomy_bm25_top = np.argsort(anatomy_bm25_scores)[::-1][:5]
print("\nAnatomy BM25 top 5:")
for rank, idx in enumerate(anatomy_bm25_top, start=1):
    print(f"  rank {rank}: chunk_index={idx} score={anatomy_bm25_scores[idx]:.3f}")

print()
print("=" * 70)
print("STEP 2: What does FAISS think, on EACH index?")
print("=" * 70)
query_embedding = retrieve.model.encode([QUERY])
query_embedding = np.array(query_embedding).astype("float32")

clinical_distances, clinical_indices = retrieve.clinical_index.search(query_embedding, retrieve.CANDIDATE_POOL_SIZE)
print("Clinical FAISS top1 distance:", clinical_distances[0][0])
print("Clinical threshold:", retrieve.CLINICAL_RELEVANCE_THRESHOLD)
print("Clinical relevant (passes gate)?", clinical_distances[0][0] <= retrieve.CLINICAL_RELEVANCE_THRESHOLD)
print("Clinical FAISS top 10 indices:", list(clinical_indices[0][:10]))
for idx in clinical_indices[0][:5]:
    if idx >= 0:
        print(f"    idx={idx} source={retrieve.clinical_metadata[idx].get('source')}")

anatomy_distances, anatomy_indices = retrieve.anatomy_index.search(query_embedding, retrieve.CANDIDATE_POOL_SIZE)
print()
print("Anatomy FAISS top1 distance:", anatomy_distances[0][0])
print("Anatomy threshold:", retrieve.ANATOMY_RELEVANCE_THRESHOLD)
print("Anatomy relevant (passes gate)?", anatomy_distances[0][0] <= retrieve.ANATOMY_RELEVANCE_THRESHOLD)

print()
print("=" * 70)
print("STEP 3: What does the fused RRF comparison actually produce?")
print("=" * 70)
clinical_relevant = clinical_distances[0][0] <= retrieve.CLINICAL_RELEVANCE_THRESHOLD
anatomy_relevant = anatomy_distances[0][0] <= retrieve.ANATOMY_RELEVANCE_THRESHOLD

clinical_fused, clinical_top_rrf = (
    retrieve._hybrid_results(clinical_indices[0], retrieve.clinical_bm25, retrieve.clinical_metadata, QUERY, 5)
    if clinical_relevant else (None, -1.0)
)
anatomy_fused, anatomy_top_rrf = (
    retrieve._hybrid_results(anatomy_indices[0], retrieve.anatomy_bm25, retrieve.anatomy_metadata, QUERY, 5)
    if anatomy_relevant else (None, -1.0)
)

print("Clinical top RRF score:", clinical_top_rrf)
print("Anatomy top RRF score:", anatomy_top_rrf)
print()
print("Winner:", "CLINICAL" if clinical_top_rrf >= anatomy_top_rrf else "ANATOMY")
print("Expected: CLINICAL (this is a guideline-level risk-factor question, NICE hypertension guideline covers this)")

print()
print("=" * 70)
print("STEP 4: What does retrieve() actually return right now?")
print("=" * 70)
final_results = retrieve.retrieve(QUERY, top_k=5)
for r in final_results:
    print(f"  source={r['source']}  text={r['text'][:80]!r}")
    