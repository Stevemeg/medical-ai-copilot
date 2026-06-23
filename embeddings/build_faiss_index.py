import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss

# Project root is the parent of this file's parent directory (embeddings/ -> project root)
BASE_DIR = Path(__file__).resolve().parent.parent

# Paths
CHUNKS_FILE = BASE_DIR / "data" / "chunks.json"
VECTOR_DIR = BASE_DIR / "data" / "vector_store"

VECTOR_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------
# Embedding Model Selection
# -----------------------------------
# Set this to the embedding model to use. Each model gets its own set of
# index/metadata files (tagged with MODEL_TAG below), so switching models
# here does NOT overwrite a previously-built index -- this lets you build
# indexes from multiple models side by side and compare them with
# embeddings/compare_embeddings.py before committing to one.
#
# TESTED AND REJECTED: NeuML/pubmedbert-base-embeddings was measured
# against this exact corpus using embeddings/compare_embeddings.py.
# Despite a published benchmark advantage on PubMed-abstract-similarity
# tasks (95.62% vs MiniLM's 93.46%), it performed WORSE on this corpus:
# - Relative separation (gap between relevant/irrelevant distances, scaled
#   to each model's own distance range) was roughly 0.56x the relevant-
#   question spread for PubMedBERT, vs roughly 2.05x for MiniLM -- a much
#   tighter, riskier margin once normalized for the different raw distance
#   scales the two models produce.
# - It actively misclassified a real clinical term ("What is the SINBAD
#   classification?" -- a term that appears verbatim in the diabetic foot
#   guideline already indexed) as MORE irrelevant than several genuinely
#   off-topic questions (cooking, sports, etc.).
# This is consistent with published findings that generalist embedding
# models can outperform domain-specific ones on short-text clinical
# retrieval (arXiv:2401.01943) -- likely because PubMedBERT is tuned on
# PubMed abstracts (research-paper register), while this corpus is
# clinical guidelines (recommendation register), a different style of text
# than the benchmark it was tuned against. Do not re-attempt this swap
# without new evidence specific to this corpus.
#
# "all-MiniLM-L6-v2": general-purpose, 384-dim, fast. Confirmed via direct
# comparison to be the better choice for THIS corpus.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Short tag used in output filenames so different models' indexes don't
# collide when experimenting with alternatives. Empty string reproduces the
# original unsuffixed filenames (clinical_faiss.index, etc.) that
# retrieve.py expects for the production index. Only set this to a non-empty
# value when deliberately building a side-by-side comparison index -- and
# remember to set it back to "" afterward, or retrieve.py will keep reading
# the OLD production index while you think you're testing a new one.
MODEL_TAG = ""

# Sources to route into the ANATOMY index rather than the CLINICAL index.
# These are background/educational textbook sources -- useful for
# foundational physiology questions, but they shouldn't compete with actual
# clinical guidelines for diagnostic/treatment questions (confirmed in
# audit: this source was 64-70% of a single mixed index and occasionally
# out-competed guideline chunks for clinically-relevant queries). Add new
# source filenames here if more background/textbook material is indexed
# later -- everything not listed here goes into the clinical index.
ANATOMY_SOURCES = {
    "openstax_anatomy_physiology.pdf.txt",
}

# Load chunks
with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

clinical_chunks = [c for c in chunks if c["source"] not in ANATOMY_SOURCES]
anatomy_chunks = [c for c in chunks if c["source"] in ANATOMY_SOURCES]

print(f"Embedding model: {EMBEDDING_MODEL_NAME}")
print(f"Clinical chunks: {len(clinical_chunks)}")
print(f"Anatomy chunks: {len(anatomy_chunks)}")

if not clinical_chunks:
    raise ValueError(
        "No chunks matched the clinical index -- check ANATOMY_SOURCES "
        "isn't accidentally matching every source filename."
    )
if not anatomy_chunks:
    print(
        "Warning: no chunks matched ANATOMY_SOURCES. The anatomy index "
        "will be empty. This is fine if you've intentionally removed "
        "those source files, but check ANATOMY_SOURCES if not."
    )

# Load embedding model (shared across both indexes -- same embedding space,
# just partitioned into two separate FAISS indexes)
model = SentenceTransformer(EMBEDDING_MODEL_NAME)


def build_index(chunk_list, index_filename, meta_filename, label):
    if not chunk_list:
        print(f"Skipping {label} index -- no chunks to embed.")
        return

    texts = [c["text"] for c in chunk_list]

    print(f"Generating embeddings for {label} index ({len(texts)} chunks)...")
    embeddings = model.encode(texts, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    index_file = VECTOR_DIR / index_filename
    meta_file = VECTOR_DIR / meta_filename

    faiss.write_index(index, str(index_file))
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(chunk_list, f, indent=2)

    print(f"{label} FAISS index saved to: {index_file}")
    print(f"{label} metadata saved to: {meta_file}")
    print(f"{label} total vectors indexed: {index.ntotal}")
    print(f"{label} embedding dimension: {dimension}")
    print()


def tagged_filename(base_name: str, extension: str) -> str:
    """
    Builds an output filename, inserting "_{MODEL_TAG}" only when MODEL_TAG
    is non-empty. With MODEL_TAG = "", this reproduces the original
    unsuffixed filename exactly (e.g. "clinical_faiss.index"), not
    "clinical_faiss_.index".
    """
    if MODEL_TAG:
        return f"{base_name}_{MODEL_TAG}.{extension}"
    return f"{base_name}.{extension}"


build_index(
    clinical_chunks,
    tagged_filename("clinical_faiss", "index"),
    tagged_filename("clinical_metadata", "json"),
    "Clinical",
)
build_index(
    anatomy_chunks,
    tagged_filename("anatomy_faiss", "index"),
    tagged_filename("anatomy_metadata", "json"),
    "Anatomy",
)