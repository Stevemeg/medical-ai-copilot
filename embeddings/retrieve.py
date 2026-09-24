import json
import re
import numpy as np
import faiss

from pathlib import Path
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from backend.knowledge_models import RetrievalPolicy
from backend.source_registry import SourceRegistry, StaleArtifact
from backend.index_provenance import validate_chunks_manifest, validate_manifest

# -----------------------------------
# Project Paths
# -----------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

VECTOR_DIR = BASE_DIR / "data" / "vector_store"

CLINICAL_INDEX_FILE = VECTOR_DIR / "clinical_faiss.index"
CLINICAL_META_FILE = VECTOR_DIR / "clinical_metadata.json"

ANATOMY_INDEX_FILE = VECTOR_DIR / "anatomy_faiss.index"
ANATOMY_META_FILE = VECTOR_DIR / "anatomy_metadata.json"

# -----------------------------------
# Validate Files Exist
# -----------------------------------

if not CLINICAL_INDEX_FILE.exists():
    raise FileNotFoundError(
        f"Clinical FAISS index not found: {CLINICAL_INDEX_FILE}. "
        f"Run embeddings/build_faiss_index.py first."
    )

if not CLINICAL_META_FILE.exists():
    raise FileNotFoundError(
        f"Clinical metadata file not found: {CLINICAL_META_FILE}"
    )

# The reference index is optional when no reference sources are registered.
ANATOMY_INDEX_AVAILABLE = ANATOMY_INDEX_FILE.exists() and ANATOMY_META_FILE.exists()

# -----------------------------------
# Load FAISS Indexes
# -----------------------------------

registry = SourceRegistry()
validate_chunks_manifest(BASE_DIR / "data/chunks.json", registry)
validate_manifest(CLINICAL_INDEX_FILE, CLINICAL_META_FILE, registry)
clinical_index = faiss.read_index(str(CLINICAL_INDEX_FILE))

with open(CLINICAL_META_FILE, "r", encoding="utf-8") as f:
    clinical_metadata = json.load(f)

if ANATOMY_INDEX_AVAILABLE:
    validate_manifest(ANATOMY_INDEX_FILE, ANATOMY_META_FILE, registry)
    anatomy_index = faiss.read_index(str(ANATOMY_INDEX_FILE))
    with open(ANATOMY_META_FILE, "r", encoding="utf-8") as f:
        anatomy_metadata = json.load(f)
else:
    anatomy_index = None
    anatomy_metadata = []

if clinical_index.ntotal != len(clinical_metadata) or (anatomy_index is not None and anatomy_index.ntotal != len(anatomy_metadata)):
    raise StaleArtifact("FAISS vector count differs from evidence metadata")


def _policy_index(policy: RetrievalPolicy):
    """Build a compact view so blocked vectors cannot affect ranking or gating."""
    source_pairs = [(clinical_index, clinical_metadata)]
    if anatomy_index is not None:
        source_pairs.append((anatomy_index, anatomy_metadata))
    selected = [(index, offset, registry.enrich(chunk))
                for index, metadata in source_pairs
                for offset, chunk in enumerate(metadata)
                if registry.eligible(chunk, policy)]
    if not selected:
        return None, [], None
    filtered = faiss.IndexFlatL2(source_pairs[0][0].d)
    vectors = np.stack([index.reconstruct(offset) for index, offset, _ in selected]).astype("float32")
    filtered.add(vectors)
    metadata = [chunk for _, _, chunk in selected]
    return filtered, metadata, build_bm25_index(metadata)

# -----------------------------------
# Load Embedding Model
# -----------------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------------
# Build BM25 Indexes
# -----------------------------------
# BM25 is a keyword/lexical search algorithm, complementing FAISS's semantic
# (embedding-based) search. Added because vector search alone showed weaker
# signal on exact-term queries during embedding-model testing (e.g. "ACEi
# vs ARB for hypertension" scored notably lower than plain-English clinical
# questions; "SINBAD classification" -- a real term from the diabetic foot
# guideline -- was misclassified entirely by one embedding model). BM25
# catches exact vocabulary matches like these regardless of how well they
# embed semantically.
#
# Built once at module load time (like the FAISS indexes above), not
# per-query -- confirmed via timing test this takes about 1 second to build
# for a ~2000-chunk corpus, comparable to loading the embedding model.


def simple_tokenize(text: str) -> list[str]:
    """
    Lowercase, alphanumeric-only tokenization. Deliberately simple (no
    stemming, no stopword removal) -- BM25's own IDF weighting already
    naturally downweights common words like "the", "is", "a" since they
    appear in nearly every document, confirmed empirically: a fully
    irrelevant query like "how do I bake a chocolate cake" scored exactly
    0.0 against medical-text documents in testing, with no stopword-overlap
    false positives observed.
    """
    return re.findall(r"[a-z0-9]+", text.lower())


def build_bm25_index(metadata: list[dict]):
    if not metadata:
        return None
    tokenized_corpus = [simple_tokenize(chunk.get("text", "")) for chunk in metadata]
    return BM25Okapi(tokenized_corpus)


# -----------------------------------
# Relevance thresholds for policy-filtered views
# -----------------------------------
# Rechecked after excluding obsolete/non-guideline vectors: representative
# eligible clinical queries measured 0.58-1.61; tested irrelevant queries
# measured 1.91-1.95. Recalibrate whenever this small corpus changes.
CLINICAL_RELEVANCE_THRESHOLD = 1.70
ANATOMY_RELEVANCE_THRESHOLD = 1.38

# RRF ranks FAISS and BM25 candidates within the eligible evidence view.
RRF_K = 60

# Within RRF, weight BM25 contributions higher than FAISS contributions.
# This directly reflects what testing showed: exact-term queries (drug
# abbreviations, guideline codes) are where vector search is weakest, so
# when BM25 and FAISS disagree, trust the exact match somewhat more.
RRF_BM25_WEIGHT = 1.0
RRF_FAISS_WEIGHT = 0.7

# -----------------------------------
# Reciprocal Rank Fusion
# -----------------------------------

def _rrf_fuse(faiss_indices, bm25_ranking, top_k, weight_faiss=RRF_FAISS_WEIGHT, weight_bm25=RRF_BM25_WEIGHT):
    """
    Fuses a FAISS ranking (list of chunk indices, best first) and a BM25
    ranking (list of chunk indices, best first) into one ranked list using
    weighted Reciprocal Rank Fusion:

        RRF(d) = weight_faiss / (RRF_K + rank_faiss(d))
               + weight_bm25  / (RRF_K + rank_bm25(d))

    where rank is the 1-indexed position in each list, or treated as
    absent (contributing 0) if the chunk doesn't appear in that list at
    all. This is the standard RRF formula (Cormack, Clarke & Buettcher,
    SIGIR 2009), with the addition of per-method weights -- a common
    variant when one retrieval method is known to be more trustworthy for
    a given corpus. Weighting BM25 higher reflects what testing showed:
    exact-term queries (drug abbreviations, guideline codes) are where
    vector search is weakest on this corpus.

    Operating purely on RANKS (not raw scores) sidesteps the scale-
    incompatibility problem between FAISS's unbounded L2 distances and
    BM25's unbounded keyword-overlap scores -- no normalization needed.

    Returns ranked indices and the top fused score for diagnostics.
    """
    rrf_scores = {}

    for rank, idx in enumerate(faiss_indices, start=1):
        rrf_scores[idx] = rrf_scores.get(idx, 0.0) + weight_faiss / (RRF_K + rank)

    for rank, idx in enumerate(bm25_ranking, start=1):
        rrf_scores[idx] = rrf_scores.get(idx, 0.0) + weight_bm25 / (RRF_K + rank)

    if not rrf_scores:
        return [], -1.0

    ranked = sorted(rrf_scores.keys(), key=lambda idx: rrf_scores[idx], reverse=True)
    top_score = rrf_scores[ranked[0]]
    return ranked[:top_k], top_score


def _bm25_top_indices(bm25_index, query: str, n: int):
    if bm25_index is None:
        return []
    scores = bm25_index.get_scores(simple_tokenize(query))
    # argsort descending, take top n. A BM25 score of 0 means no keyword
    # overlap at all -- exclude those rather than letting zero-relevance
    # chunks pad out the ranking (confirmed empirically: fully irrelevant
    # queries score exactly 0.0 against every chunk, so this cleanly
    # excludes them from contributing to the fused ranking entirely).
    ranked_idx = np.argsort(scores)[::-1]
    return [int(i) for i in ranked_idx[:n] if scores[i] > 0]


# -----------------------------------
# Retrieval Function
# -----------------------------------

# Fetch a wider candidate pool than the final top_k from each retrieval
# method before fusing, so BM25 has room to surface a chunk that FAISS's
# narrower top-k alone would have missed entirely -- not just reorder
# whatever FAISS already returned.
CANDIDATE_POOL_SIZE = 20


def retrieve(query: str, top_k: int = 5, policy: RetrievalPolicy = RetrievalPolicy.CURRENT_CLINICAL):
    """Hybrid retrieval within an explicit evidence and lifecycle policy."""
    policy = RetrievalPolicy(policy)
    index, metadata, bm25 = policy_indexes[policy]
    if index is None or top_k <= 0:
        return []
    embedding = np.array(model.encode([query])).astype("float32")
    distances, candidates = index.search(embedding, CANDIDATE_POOL_SIZE)
    threshold = ANATOMY_RELEVANCE_THRESHOLD if policy is RetrievalPolicy.REFERENCE else CLINICAL_RELEVANCE_THRESHOLD
    if distances[0][0] > threshold:
        return []
    results, _ = _hybrid_results(candidates[0], bm25, metadata, query, top_k)
    return results


policy_indexes = {policy: _policy_index(policy) for policy in RetrievalPolicy}


def _hybrid_results(faiss_candidate_indices, bm25_index, metadata, query, top_k):
    bm25_candidate_indices = _bm25_top_indices(bm25_index, query, CANDIDATE_POOL_SIZE)
    # FAISS pads results with -1 when the requested top_k (CANDIDATE_POOL_SIZE
    # here) exceeds the number of vectors actually in the index. Python
    # silently treats -1 as a valid "last element" index rather than raising
    # an error, which caused a real bug found in testing: a small index
    # produced duplicate/wrong results because -1 resolved to the last
    # chunk in metadata instead of being recognized as "no result."
    valid_faiss_indices = [int(idx) for idx in faiss_candidate_indices if idx != -1]
    fused_indices, top_rrf_score = _rrf_fuse(valid_faiss_indices, bm25_candidate_indices, top_k)
    return _build_results(fused_indices, metadata), top_rrf_score


def _build_results(indices, metadata):
    results = []

    for idx in indices:

        # FAISS uses -1 as a "no result" sentinel when fewer matches exist
        # than requested. idx < len(metadata) alone doesn't exclude this --
        # Python's negative indexing makes metadata[-1] silently resolve to
        # the LAST element rather than erroring, which caused a real
        # duplicate-result bug found in testing. Explicitly require idx >= 0.
        if 0 <= idx < len(metadata):

            chunk = metadata[idx]

            results.append(chunk)

    return results

# -----------------------------------
# CLI Testing
# -----------------------------------

if __name__ == "__main__":

    while True:

        q = input("\nAsk a medical question (or 'exit'): ")

        if q.lower() == "exit":
            break

        results = retrieve(q)

        for i, res in enumerate(results, 1):

            print(f"\n--- Result {i} ({res['source']}) ---")

            print(res["text"][:800], "...")
