import json
import numpy as np
import faiss

from pathlib import Path
from sentence_transformers import SentenceTransformer

# -----------------------------------
# Project Paths
# -----------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

VECTOR_DIR = BASE_DIR / "data" / "vector_store"

INDEX_FILE = VECTOR_DIR / "faiss.index"
META_FILE = VECTOR_DIR / "metadata.json"

# -----------------------------------
# Validate Files Exist
# -----------------------------------

if not INDEX_FILE.exists():
    raise FileNotFoundError(
        f"FAISS index not found: {INDEX_FILE}"
    )

if not META_FILE.exists():
    raise FileNotFoundError(
        f"Metadata file not found: {META_FILE}"
    )

# -----------------------------------
# Load FAISS Index
# -----------------------------------

index = faiss.read_index(str(INDEX_FILE))

# -----------------------------------
# Load Metadata
# -----------------------------------

with open(META_FILE, "r", encoding="utf-8") as f:
    metadata = json.load(f)

# -----------------------------------
# Load Embedding Model
# -----------------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------------
# Retrieval Function
# -----------------------------------

def retrieve(query: str, top_k: int = 5):

    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    distances, indices = index.search(query_embedding, top_k)

    results = []

    for idx in indices[0]:

        if idx < len(metadata):

            chunk = metadata[idx]

            results.append({
                "text": chunk.get("text", ""),
                "source": chunk.get("source", "Unknown Source")
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