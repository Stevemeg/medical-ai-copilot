"""
Diagnostic script — NOT part of the app. Compares the existing MiniLM
clinical index against a newly-built PubMedBERT clinical index on the same
real questions, so we can see an actual side-by-side before deciding
whether to commit to the embedding-model swap.

Usage:
    python embeddings/compare_embeddings.py

Requires both clinical_faiss.index (MiniLM, built earlier) and
clinical_faiss_pubmedbert.index (built by running build_faiss_index.py
with EMBEDDING_MODEL_NAME set to the PubMedBERT model) to already exist.

This does not modify any index or metadata file. It just prints numbers.
"""

import json
import numpy as np
import faiss

from pathlib import Path
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent
VECTOR_DIR = BASE_DIR / "data" / "vector_store"

MINILM_INDEX_FILE = VECTOR_DIR / "clinical_faiss.index"
MINILM_META_FILE = VECTOR_DIR / "clinical_metadata.json"

PUBMEDBERT_INDEX_FILE = VECTOR_DIR / "clinical_faiss_pubmedbert.index"
PUBMEDBERT_META_FILE = VECTOR_DIR / "clinical_metadata_pubmedbert.json"

# Same question set used for clinical-index calibration earlier, so results
# are directly comparable to the existing 1.12 threshold baseline.
RELEVANT_QUESTIONS = [
    "What are the symptoms of type 2 diabetes?",
    "How is diabetes managed according to clinical guidelines?",
    "What are the risk factors for hypertension?",
    "How is hypertension diagnosed?",
    "What blood pressure target should be used for adults with diabetes?",
    "When should statins be offered for cardiovascular risk reduction?",
    "How should a diabetic foot ulcer be managed?",
    "What are the risk factors for chronic disease?",
]

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

# A few harder/ambiguous questions worth checking specifically -- these are
# cases where the two models might disagree, since they're more subtle than
# the clean relevant/irrelevant split above.
HARD_QUESTIONS = [
    "Explain insulin resistance.",  # the boundary case from earlier testing
    "What is the SINBAD classification?",  # a specific clinical term/acronym
    "ACEi vs ARB for hypertension",  # abbreviation-heavy clinical shorthand
]


def load_index(index_file, meta_file):
    if not index_file.exists() or not meta_file.exists():
        return None, None
    index = faiss.read_index(str(index_file))
    with open(meta_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata


def measure(label, questions, index, metadata, model):
    print(f"\n  {label}")
    dists = []
    for q in questions:
        emb = model.encode([q]).astype("float32")
        d, i = index.search(emb, 1)
        top1 = d[0][0]
        source = metadata[i[0][0]].get("source", "?")
        dists.append(top1)
        print(f"    [{top1:7.3f}]  {q!r:55s} -> {source}")
    arr = np.array(dists)
    print(f"    range: {arr.min():.3f} to {arr.max():.3f}, mean: {arr.mean():.3f}")
    return arr


minilm_index, minilm_meta = load_index(MINILM_INDEX_FILE, MINILM_META_FILE)
pubmedbert_index, pubmedbert_meta = load_index(PUBMEDBERT_INDEX_FILE, PUBMEDBERT_META_FILE)

if minilm_index is None:
    print("MiniLM clinical index not found -- nothing to compare against.")
    raise SystemExit(1)

if pubmedbert_index is None:
    print("PubMedBERT clinical index not found.")
    print("Run build_faiss_index.py first (with EMBEDDING_MODEL_NAME set to")
    print("NeuML/pubmedbert-base-embeddings) to generate it.")
    raise SystemExit(1)

print("Loading models (this downloads PubMedBERT on first run, ~440MB)...")
minilm_model = SentenceTransformer("all-MiniLM-L6-v2")
pubmedbert_model = SentenceTransformer("NeuML/pubmedbert-base-embeddings")

print("\n" + "=" * 70)
print("RELEVANT QUESTIONS")
print("=" * 70)
minilm_rel = measure("MiniLM", RELEVANT_QUESTIONS, minilm_index, minilm_meta, minilm_model)
pubmed_rel = measure("PubMedBERT", RELEVANT_QUESTIONS, pubmedbert_index, pubmedbert_meta, pubmedbert_model)

print("\n" + "=" * 70)
print("IRRELEVANT QUESTIONS")
print("=" * 70)
minilm_irr = measure("MiniLM", IRRELEVANT_QUESTIONS, minilm_index, minilm_meta, minilm_model)
pubmed_irr = measure("PubMedBERT", IRRELEVANT_QUESTIONS, pubmedbert_index, pubmedbert_meta, pubmedbert_model)

print("\n" + "=" * 70)
print("HARD / BOUNDARY QUESTIONS (most informative for deciding)")
print("=" * 70)
measure("MiniLM", HARD_QUESTIONS, minilm_index, minilm_meta, minilm_model)
measure("PubMedBERT", HARD_QUESTIONS, pubmedbert_index, pubmedbert_meta, pubmedbert_model)

print("\n" + "=" * 70)
print("SUMMARY -- separation quality (bigger gap = cleaner relevance signal)")
print("=" * 70)
minilm_gap = minilm_irr.min() - minilm_rel.max()
pubmed_gap = pubmed_irr.min() - pubmed_rel.max()
print(f"MiniLM gap (irrelevant_min - relevant_max):     {minilm_gap:.3f}")
print(f"PubMedBERT gap (irrelevant_min - relevant_max): {pubmed_gap:.3f}")
print()
if pubmed_gap > minilm_gap:
    print("PubMedBERT shows a WIDER separation gap on this corpus -- supports switching.")
elif pubmed_gap < minilm_gap:
    print("MiniLM shows a WIDER separation gap on this corpus -- switching may not help,")
    print("consistent with published findings that generalist models sometimes beat")
    print("domain-specific ones on short-text clinical retrieval.")
else:
    print("No meaningful difference in separation quality on this corpus.")
print()
print("Also check the HARD QUESTIONS results above by eye -- a wider average gap")
print("doesn't guarantee better behavior on specific edge cases like abbreviation")
print("handling or boundary topics, which matter more in practice than the average.")