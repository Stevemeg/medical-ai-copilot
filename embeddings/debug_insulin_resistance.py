"""
Diagnostic script -- NOT part of the app. Traces what happens for the
"explain insulin resistance" query against your real indexes, after the
RRF_K change from 60 to 10, to confirm this previously-correct case (which
should route to ANATOMY) hasn't regressed.

Usage:
    python embeddings/debug_insulin_resistance.py
"""

import numpy as np

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import retrieve  # reuse the real module's loaded indexes, BM25, thresholds, etc.

QUERY = "explain insulin resistance"

print("=" * 70)
print("STEP 1: What does BM25 think, on EACH index?")
print("=" * 70)
clinical_bm25_scores = retrieve.clinical_bm25.get_scores(retrieve.simple_tokenize(QUERY))
clinical_bm25_top = np.argsort(clinical_bm25_scores)[::-1][:5]
print("Clinical BM25 top 5:")
for rank, idx in enumerate(clinical_bm25_top, start=1):
    print(f"  rank {rank}: chunk_index={idx} score={clinical_bm25_scores[idx]:.3f}")

if retrieve.anatomy_bm25 is not None:
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

anatomy_distances, anatomy_indices = retrieve.anatomy_index.search(query_embedding, retrieve.CANDIDATE_POOL_SIZE)
print()
print("Anatomy FAISS top1 distance:", anatomy_distances[0][0])
print("Anatomy threshold:", retrieve.ANATOMY_RELEVANCE_THRESHOLD)
print("Anatomy relevant (passes gate)?", anatomy_distances[0][0] <= retrieve.ANATOMY_RELEVANCE_THRESHOLD)

print()
print("=" * 70)
print("STEP 3: What does the fused RRF comparison actually produce NOW (RRF_K=10)?")
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
print("Expected (per earlier verified testing): ANATOMY")

print()
print("=" * 70)
print("STEP 4: What does retrieve() actually return right now?")
print("=" * 70)
final_results = retrieve.retrieve(QUERY, top_k=3)
for r in final_results:
    print(f"  source={r['source']}  text={r['text'][:80]!r}")