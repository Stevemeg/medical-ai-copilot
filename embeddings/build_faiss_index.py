import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss

# Paths
CHUNKS_FILE = Path(r"C:\medical-ai-copilot\data\chunks.json")
VECTOR_DIR = Path(r"C:\medical-ai-copilot\data\vector_store")

VECTOR_DIR.mkdir(parents=True, exist_ok=True)

INDEX_FILE = VECTOR_DIR / "faiss.index"
META_FILE = VECTOR_DIR / "metadata.json"

# Load chunks
with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

texts = [c["text"] for c in chunks]

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Generate embeddings
print("Generating embeddings...")
embeddings = model.encode(texts, show_progress_bar=True)

embeddings = np.array(embeddings).astype("float32")

# Build FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Save index
faiss.write_index(index, str(INDEX_FILE))

# Save metadata
with open(META_FILE, "w", encoding="utf-8") as f:
    json.dump(chunks, f, indent=2)

print(f"FAISS index saved to: {INDEX_FILE}")
print(f"Metadata saved to: {META_FILE}")
print(f"Total vectors indexed: {index.ntotal}")
