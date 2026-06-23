"""
Diagnostic script — NOT part of the app. Run this once locally to see real
L2 distance numbers from your own indexes, so we can pick relevance
thresholds based on actual data instead of guessed numbers.

Usage:
    python embeddings/calibrate_threshold.py

This does not modify retrieve.py or any other file. It just prints numbers.

Tests the CLINICAL and ANATOMY indexes separately, since splitting the
corpus changes the distance distribution in each -- a threshold calibrated
against one index is not valid for the other, and neither is valid for the
old single mixed index this corpus used to be.

Also calibrates a BM25 score floor, separate from the FAISS distance
threshold. This matters because of a real, confirmed failure: "What is the
SINBAD classification?" has a clinical FAISS distance of 1.27, ABOVE the
1.12 threshold, so the clinical index was being excluded entirely before
BM25 ever got a chance to weigh in -- even though BM25 found an extremely
strong, exclusive match (score 14.36) for the exact term. The relevance
gate needs to pass an index through if EITHER FAISS clears its threshold
OR BM25 finds a strong enough exact-term match, so this calibration
measures what "strong enough" actually means on real data.
"""

import json
import re
import numpy as np
import faiss

from pathlib import Path
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

BASE_DIR = Path(__file__).resolve().parent.parent
VECTOR_DIR = BASE_DIR / "data" / "vector_store"

CLINICAL_INDEX_FILE = VECTOR_DIR / "clinical_faiss.index"
CLINICAL_META_FILE = VECTOR_DIR / "clinical_metadata.json"
ANATOMY_INDEX_FILE = VECTOR_DIR / "anatomy_faiss.index"
ANATOMY_META_FILE = VECTOR_DIR / "anatomy_metadata.json"

model = SentenceTransformer("all-MiniLM-L6-v2")


def simple_tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


# Questions a CLINICAL guideline corpus should answer well -- diagnostic,
# treatment, and management questions. Deliberately NOT anatomy/physiology
# questions, since those now belong to a separate index and a clinical
# guideline corpus has no reason to answer them well.
CLINICAL_RELEVANT_QUESTIONS = [
    "What are the symptoms of type 2 diabetes?",
    "How is diabetes managed according to clinical guidelines?",
    "What are the risk factors for hypertension?",
    "How is hypertension diagnosed?",
    "What blood pressure target should be used for adults with diabetes?",
    "When should statins be offered for cardiovascular risk reduction?",
    "How should a diabetic foot ulcer be managed?",
    "What are the risk factors for chronic disease?",
]

# Questions an ANATOMY/physiology textbook should answer well, but a
# clinical guideline corpus has no reason to cover in depth.
ANATOMY_RELEVANT_QUESTIONS = [
    "What is the function of the pancreas?",
    "Explain insulin resistance.",
    "How does the endocrine system regulate blood sugar?",
    "What is the structure of a nephron?",
    "Describe the anatomy of the heart.",
    "What is the role of the liver in metabolism?",
    "How do alpha and beta cells differ?",
    "What is homeostasis?",
]

# Questions that should be irrelevant to BOTH indexes.
IRRELEVANT_QUESTIONS = [
    "What's the best programming language for web development?",
    "Who won the last World Cup?",
    "How do I bake a chocolate cake?",
    "What's the capital of France?",
    "Explain how a car engine works.",
    "What is the stock price of Apple today?",
    "How do I fix a leaking faucet?",
    "What's a good recipe for pasta carbonara?",
]

# Exact-term clinical questions that FAISS alone has been shown to score
# weakly on (general semantic embedding doesn't represent abbreviations and
# scoring-system names as strongly as plain-English clinical phrasing), but
# that should match an indexed document EXACTLY via keyword search.
EXACT_TERM_QUESTIONS = [
    "What is the SINBAD classification?",
    "ACEi vs ARB for hypertension",
    "What is NG136?",
    "What is NG19?",
]


def load_index(index_file, meta_file):
    if not index_file.exists() or not meta_file.exists():
        return None, None
    index = faiss.read_index(str(index_file))
    with open(meta_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata


def build_bm25(metadata):
    if not metadata:
        return None
    tokenized = [simple_tokenize(c.get("text", "")) for c in metadata]
    return BM25Okapi(tokenized)


def show_distances(label, questions, index, metadata):
    print(f"\n{'=' * 60}")
    print(f"{label}")
    print(f"{'=' * 60}")
    all_top1 = []
    for q in questions:
        query_embedding = model.encode([q]).astype("float32")
        distances, indices = index.search(query_embedding, 5)
        top1_dist = distances[0][0]
        top1_source = metadata[indices[0][0]].get("source", "?")
        all_top1.append(top1_dist)
        print(f"  [{top1_dist:7.2f}]  {q!r:60s}  -> {top1_source}")
    arr = np.array(all_top1)
    print(f"\n  top-1 distance stats for this group:")
    print(f"  min={arr.min():.2f}  max={arr.max():.2f}  mean={arr.mean():.2f}  median={np.median(arr):.2f}")
    return arr


def show_bm25_scores(label, questions, bm25_index):
    print(f"\n{'=' * 60}")
    print(f"{label}")
    print(f"{'=' * 60}")
    all_top1 = []
    for q in questions:
        scores = bm25_index.get_scores(simple_tokenize(q))
        top1 = float(np.max(scores))
        all_top1.append(top1)
        print(f"  [{top1:7.3f}]  {q!r:60s}")
    arr = np.array(all_top1)
    print(f"\n  top-1 BM25 score stats for this group:")
    print(f"  min={arr.min():.3f}  max={arr.max():.3f}  mean={arr.mean():.3f}  median={np.median(arr):.3f}")
    return arr


def calibrate_index(index_name, index, metadata, relevant_questions):
    if index is None:
        print(f"\n{index_name} index not found -- skipping. Run build_faiss_index.py first.")
        return

    relevant_dists = show_distances(
        f"{index_name.upper()} INDEX -- relevant questions (FAISS)", relevant_questions, index, metadata
    )
    irrelevant_dists = show_distances(
        f"{index_name.upper()} INDEX -- irrelevant questions (FAISS)", IRRELEVANT_QUESTIONS, index, metadata
    )

    print(f"\n{'=' * 60}")
    print(f"{index_name.upper()} INDEX FAISS SUMMARY")
    print(f"{'=' * 60}")
    print(f"Relevant group   — top-1 distance range: {relevant_dists.min():.2f} to {relevant_dists.max():.2f}")
    print(f"Irrelevant group — top-1 distance range: {irrelevant_dists.min():.2f} to {irrelevant_dists.max():.2f}")

    if relevant_dists.max() < irrelevant_dists.min():
        suggested = (relevant_dists.max() + irrelevant_dists.min()) / 2
        print(f"\nClean separation found. Suggested {index_name} FAISS threshold: {suggested:.2f}")
    else:
        print(f"\nThe two groups overlap for the {index_name} index -- there's no single")
        print("distance value that cleanly separates them. Look at the numbers above")
        print("together and pick a reasonable cutoff, possibly accepting some")
        print("false positives or false negatives at the boundary.")

    # BM25 calibration -- only meaningful for the clinical index in practice
    # (exact-term clinical jargon is the use case), but run for both so the
    # numbers are directly comparable.
    bm25_index = build_bm25(metadata)
    if bm25_index is None:
        return

    irrelevant_bm25 = show_bm25_scores(f"{index_name.upper()} INDEX -- irrelevant questions (BM25)", IRRELEVANT_QUESTIONS, bm25_index)
    exact_term_bm25 = show_bm25_scores(f"{index_name.upper()} INDEX -- exact-term questions (BM25)", EXACT_TERM_QUESTIONS, bm25_index)

    print(f"\n{'=' * 60}")
    print(f"{index_name.upper()} INDEX BM25 SUMMARY")
    print(f"{'=' * 60}")
    print(f"Irrelevant group  — top-1 BM25 score range: {irrelevant_bm25.min():.3f} to {irrelevant_bm25.max():.3f}")
    print(f"Exact-term group  — top-1 BM25 score range: {exact_term_bm25.min():.3f} to {exact_term_bm25.max():.3f}")
    if exact_term_bm25.min() > irrelevant_bm25.max():
        suggested_bm25 = (exact_term_bm25.min() + irrelevant_bm25.max()) / 2
        print(f"\nClean separation found. Suggested {index_name} BM25 floor: {suggested_bm25:.2f}")
    else:
        print(f"\nOverlap between irrelevant and exact-term BM25 scores for {index_name} --")
        print("look at the numbers above directly to pick a reasonable floor.")


clinical_index, clinical_metadata = load_index(CLINICAL_INDEX_FILE, CLINICAL_META_FILE)
anatomy_index, anatomy_metadata = load_index(ANATOMY_INDEX_FILE, ANATOMY_META_FILE)

calibrate_index("clinical", clinical_index, clinical_metadata, CLINICAL_RELEVANT_QUESTIONS)
calibrate_index("anatomy", anatomy_index, anatomy_metadata, ANATOMY_RELEVANT_QUESTIONS)