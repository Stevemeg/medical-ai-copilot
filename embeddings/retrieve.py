import json
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Paths
VECTOR_DIR = Path(r"C:\medical-ai-copilot\data\vector_store")
INDEX_FILE = VECTOR_DIR / "faiss.index"
META_FILE = VECTOR_DIR / "metadata.json"

# Load FAISS index
index = faiss.read_index(str(INDEX_FILE))

# Load metadata
with open(META_FILE, "r", encoding="utf-8") as f:
    metadata = json.load(f)

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

def retrieve(query: str, top_k: int = 5):
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    distances, indices = index.search(query_embedding, top_k)

    results = []
    for idx in indices[0]:
        chunk = metadata[idx]
        results.append({
            "text": chunk["text"],
            "source": chunk["source"]
        })

    return results

if __name__ == "__main__":
    while True:
        q = input("\nAsk a medical question (or 'exit'): ")
        if q.lower() == "exit":
            break

        results = retrieve(q)
        for i, res in enumerate(results, 1):
            print(f"\n--- Result {i} ({res['source']}) ---")
            print(res["text"][:800], "...")
