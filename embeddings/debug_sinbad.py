"""
Diagnostic script -- NOT part of the app. Traces exactly what happens for
the "SINBAD" query against your real indexes, step by step, so we can see
where the routing decision is actually going wrong instead of guessing.

Usage:
    python embeddings/debug_sinbad.py
"""

import json
import numpy as np
import faiss

from pathlib import Path
from sentence_transformers import SentenceTransformer

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import retrieve  # reuse the real module's loaded indexes, BM25, thresholds, etc.

QUERY = "What is the SINBAD classification?"

print("=" * 70)
print("STEP 1: Does 'sinbad' appear in any clinical chunk at all?")
print("=" * 70)
matches = [
    (i, chunk["source"], chunk.get("page_start"))
    for i, chunk in enumerate(retrieve.clinical_metadata)
    if "sinbad" in chunk.get("text", "").lower()
]
print(f"Chunks containing 'sinbad' (case-insensitive): {len(matches)}")
for idx, source, page in matches[:10]:
    print(f"  chunk_index={idx}  source={source}  page_start={page}")

print()
print("=" * 70)
print("STEP 2: What does BM25 think, directly?")
print("=" * 70)
bm25_scores = retrieve.clinical_bm25.get_scores(retrieve.simple_tokenize(QUERY))
top10 = np.argsort(bm25_scores)[::-1][:10]
for rank, idx in enumerate(top10, start=1):
    score = bm25_scores[idx]
    text_preview = retrieve.clinical_metadata[idx]["text"][:80]
    print(f"  BM25 rank {rank}: chunk_index={idx} score={score:.3f}  {text_preview!r}")

print()
print("=" * 70)
print("STEP 3: What does FAISS think, directly?")
print("=" * 70)
query_embedding = retrieve.model.encode([QUERY])
query_embedding = np.array(query_embedding).astype("float32")

clinical_distances, clinical_indices = retrieve.clinical_index.search(query_embedding, retrieve.CANDIDATE_POOL_SIZE)
print("Clinical FAISS top1 distance:", clinical_distances[0][0])
print("Clinical threshold:", retrieve.CLINICAL_RELEVANCE_THRESHOLD)
print("Clinical relevant (passes gate)?", clinical_distances[0][0] <= retrieve.CLINICAL_RELEVANCE_THRESHOLD)
print("Clinical FAISS top 10 indices:", list(clinical_indices[0][:10]))

if retrieve.anatomy_index is not None:
    anatomy_distances, anatomy_indices = retrieve.anatomy_index.search(query_embedding, retrieve.CANDIDATE_POOL_SIZE)
    print()
    print("Anatomy FAISS top1 distance:", anatomy_distances[0][0])
    print("Anatomy threshold:", retrieve.ANATOMY_RELEVANCE_THRESHOLD)
    print("Anatomy relevant (passes gate)?", anatomy_distances[0][0] <= retrieve.ANATOMY_RELEVANCE_THRESHOLD)

print()
print("=" * 70)
print("STEP 4: What does the fused RRF comparison actually produce?")
print("=" * 70)
clinical_relevant = clinical_distances[0][0] <= retrieve.CLINICAL_RELEVANCE_THRESHOLD
anatomy_relevant = anatomy_distances[0][0] <= retrieve.ANATOMY_RELEVANCE_THRESHOLD if retrieve.anatomy_index is not None else False

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
print("Decision: clinical_top_rrf >= anatomy_top_rrf ->", clinical_top_rrf >= anatomy_top_rrf)
print("This means the winner SHOULD be:", "CLINICAL" if clinical_top_rrf >= anatomy_top_rrf else "ANATOMY")

print()
print("=" * 70)
print("STEP 5: What does retrieve() actually return right now?")
print("=" * 70)
final_results = retrieve.retrieve(QUERY, top_k=5)
for r in final_results:
    print(f"  source={r['source']}  text={r['text'][:80]!r}")