import json
import re
import numpy as np
import faiss

from pathlib import Path
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

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

# The anatomy index is allowed to be missing (e.g. if ANATOMY_SOURCES in
# build_faiss_index.py matched nothing) -- in that case, fallback simply
# never has anywhere to fall back to, which is a safe degraded state, not
# a crash.
ANATOMY_INDEX_AVAILABLE = ANATOMY_INDEX_FILE.exists() and ANATOMY_META_FILE.exists()

# -----------------------------------
# Load FAISS Indexes
# -----------------------------------

clinical_index = faiss.read_index(str(CLINICAL_INDEX_FILE))

with open(CLINICAL_META_FILE, "r", encoding="utf-8") as f:
    clinical_metadata = json.load(f)

if ANATOMY_INDEX_AVAILABLE:
    anatomy_index = faiss.read_index(str(ANATOMY_INDEX_FILE))
    with open(ANATOMY_META_FILE, "r", encoding="utf-8") as f:
        anatomy_metadata = json.load(f)
else:
    anatomy_index = None
    anatomy_metadata = []

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


clinical_bm25 = build_bm25_index(clinical_metadata)
anatomy_bm25 = build_bm25_index(anatomy_metadata) if ANATOMY_INDEX_AVAILABLE else None

# -----------------------------------
# Relevance Thresholds
# -----------------------------------
# Originally calibrated against these exact indexes (MiniLM embeddings +
# IndexFlatL2, post corpus-split) using embeddings/calibrate_threshold.py:
#
# Clinical index: relevant questions measured 0.51-0.90, irrelevant
# measured 1.34-1.77 (in that calibration run). Anatomy index: relevant
# 0.73-1.25, irrelevant 1.52-1.79.
#
# CLINICAL THRESHOLD WAS LATER WIDENED FROM 1.12 TO 1.28. Real-world testing
# found a confirmed failure: "What is the SINBAD classification?" (SINBAD is
# a genuine diabetic-foot-ulcer scoring system in the indexed NICE
# guideline) had a clinical FAISS distance of 1.2715 -- ABOVE the original
# 1.12 threshold, so the clinical index was excluded entirely before BM25
# ever got a chance to weigh in, even though BM25 found an exact, correct
# match (the right chunk, ranked 1st).
#
# Three BM25-score-based attempts to fix this WITHOUT touching the FAISS
# threshold were tried and failed on real measured data:
#   1. Absolute BM25 score floor -- failed: a genuinely relevant exact-term
#      query ("What is NG136?") scored 8.92, LOWER than several confirmed-
#      irrelevant queries (9.83-18.43). No floor can separate these.
#   2. Max single-token IDF -- abandoned before implementation: a rare-but-
#      irrelevant word (e.g. "cake") gets just as high an IDF as a rare
#      medically-meaningful word, so this doesn't distinguish relevance.
#   3. BM25 rank-1-vs-rank-2 score gap -- failed: "What's the capital of
#      France?" (irrelevant) had a LARGER gap (5.43) than the genuine
#      SINBAD match (3.52).
#
# Given no BM25-score-based signal cleanly separated genuine matches from
# noise on this corpus, the actual fix widens the FAISS threshold itself
# just enough to admit the confirmed SINBAD case (1.2715), with a small
# deliberate margin (1.28), while staying below the lowest confirmed-
# irrelevant distance measured (1.32). This margin is THIN (0.04) -- if a
# new false positive shows up at the threshold boundary in practice, this
# is the place to look, and re-running calibrate_threshold.py with a wider
# irrelevant-question set is the right next step rather than nudging this
# number blindly.
#
# These thresholds gate WHETHER an index is considered AT ALL (based on
# FAISS semantic distance only) -- an index with no plausible semantic
# match is excluded regardless of any BM25 signal. WHICH index wins when
# both clear this gate is ALSO decided by raw FAISS distance (lower
# wins) -- not by RRF/BM25 score (see retrieve()'s docstring for why
# RRF score was tried for this and found to cause a different real bug).
#
# Re-calibrate if the embedding model changes, or if either index's
# underlying corpus changes significantly (e.g. ANATOMY_SOURCES in
# build_faiss_index.py is updated to include more/different source files).
CLINICAL_RELEVANCE_THRESHOLD = 1.28
ANATOMY_RELEVANCE_THRESHOLD = 1.38

# RRF smoothing constant. Reverted to the standard default of 60
# (Elasticsearch, OpenSearch, Cormack/Clarke/Buettcher SIGIR 2009).
#
# This was previously lowered to 10 to fix a cross-index SELECTION bug
# (a mediocre-but-multi-method chunk beating an exclusive BM25 rank-1
# match when comparing RRF scores ACROSS indexes). That selection logic
# has since been replaced entirely -- index selection now compares raw
# FAISS distance, not RRF score (see retrieve()'s docstring for the full
# history) -- so RRF_K no longer affects which index wins, only the
# ORDER of results within whichever index is already selected. Reverted
# to the standard value since the original reason for changing it no
# longer applies to RRF's current, narrower job. Not yet re-verified
# whether 60 vs 10 makes any practical difference for within-index
# ordering specifically -- if a within-index ranking issue surfaces,
# this is a real, isolated thing to test on its own, separate from the
# cross-index selection logic.
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

    Returns (ranked_indices, top_score) -- the top_score is needed so
    callers can compare relevance strength ACROSS indexes built from
    different FAISS distance scales, which raw FAISS distance alone
    cannot reliably do once BM25 is in the mix (see retrieve()'s
    docstring for why this comparison matters).
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


def retrieve(query: str, top_k: int = 5):
    """
    Queries both the clinical index (guidelines, literature) and the
    anatomy index (background/physiology textbook content). WHICH index's
    results are returned is decided by comparing each index's raw FAISS
    top-1 distance (lower wins) -- NOT by RRF/BM25 score. Once an index is
    selected, results WITHIN that index are produced by fusing FAISS's
    semantic ranking with BM25's keyword ranking via Reciprocal Rank
    Fusion, so exact-term queries (drug abbreviations like "ACEi",
    guideline codes like "NG28", clinical classification names like
    "SINBAD") that score weakly in embedding space but match exactly in
    keyword space still surface correctly WITHIN the chosen index.

    This two-signal split (FAISS distance for WHICH index, RRF for WHICH
    chunk within it) replaced an earlier design that used RRF's fused
    score for the cross-index decision too. That design was wrong, found
    via a real failure: for "What are the risk factors for hypertension?",
    clinical has several good-but-different hypertension-guideline chunks
    -- the chunk BM25 likes best and the chunk FAISS likes best aren't the
    SAME chunk, splitting clinical's "votes" across two candidates. This
    let anatomy's single chunk that's merely DECENT by both methods (not
    exceptional by either) out-score clinical's genuinely better content,
    purely because anatomy's much larger corpus (1982 vs 1134 chunks) had
    more chances to find something passably consistent across both
    methods. RRF's fused score measures "how strong is the single best
    chunk across two methods," not "how relevant is this index overall" --
    using it for index selection conflated two different questions.

    Raw FAISS distance doesn't have this problem: it's the same continuous
    embedding-similarity measure for every query against every index,
    directly comparable regardless of how many chunks either index has or
    how votes split within it. Verified against all three real cases
    measured during debugging: clinical distance beats anatomy's for both
    "SINBAD classification" (1.27 vs 1.35) and "risk factors for
    hypertension" (0.82 vs 0.86), while anatomy correctly beats clinical
    for "explain insulin resistance" (1.01 vs 1.08) -- raw distance
    comparison gets all three right.

    A relevance GATE based on FAISS distance against each index's
    calibrated threshold still applies first -- an index with no
    plausible semantic match at all is excluded entirely. This gate was
    previously too strict for the clinical index specifically (see
    CLINICAL_RELEVANCE_THRESHOLD's comments for that history) and was
    widened to 1.28 to admit the confirmed SINBAD case.
    """

    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    clinical_distances, clinical_candidate_indices = clinical_index.search(query_embedding, CANDIDATE_POOL_SIZE)
    clinical_top1 = clinical_distances[0][0]
    clinical_relevant = clinical_top1 <= CLINICAL_RELEVANCE_THRESHOLD

    anatomy_relevant = False
    anatomy_top1 = None
    anatomy_candidate_indices = None
    if anatomy_index is not None:
        anatomy_distances, anatomy_candidate_indices = anatomy_index.search(query_embedding, CANDIDATE_POOL_SIZE)
        anatomy_top1 = anatomy_distances[0][0]
        anatomy_relevant = anatomy_top1 <= ANATOMY_RELEVANCE_THRESHOLD

    if clinical_relevant and anatomy_relevant:
        # Both indexes have a plausible semantic match -- compare raw
        # FAISS distance (lower = more similar = wins), NOT RRF/BM25
        # score. See this function's docstring for why.
        if clinical_top1 <= anatomy_top1:
            clinical_fused, _ = _hybrid_results(clinical_candidate_indices[0], clinical_bm25, clinical_metadata, query, top_k)
            return clinical_fused
        anatomy_fused, _ = _hybrid_results(anatomy_candidate_indices[0], anatomy_bm25, anatomy_metadata, query, top_k)
        return anatomy_fused

    if clinical_relevant:
        clinical_fused, _ = _hybrid_results(clinical_candidate_indices[0], clinical_bm25, clinical_metadata, query, top_k)
        return clinical_fused

    if anatomy_relevant:
        anatomy_fused, _ = _hybrid_results(anatomy_candidate_indices[0], anatomy_bm25, anatomy_metadata, query, top_k)
        return anatomy_fused

    return []


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

            results.append({
                "text": chunk.get("text", ""),
                "source": chunk.get("source", "Unknown Source"),
                "page_start": chunk.get("page_start"),
                "page_end": chunk.get("page_end"),
            })

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