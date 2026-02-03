import json
from pathlib import Path
import nltk
import tiktoken

nltk.download("punkt")

DATA_DIR = Path(r"C:\medical-ai-copilot\data\processed")
OUT_FILE = Path(r"C:\medical-ai-copilot\data\chunks.json")

CHUNK_SIZE = 600
OVERLAP = 100

tokenizer = tiktoken.get_encoding("cl100k_base")

def tokenize(text):
    return tokenizer.encode(text)

def detokenize(tokens):
    return tokenizer.decode(tokens)

all_chunks = []

for txt_file in DATA_DIR.glob("*.txt"):
    with open(txt_file, "r", encoding="utf-8") as f:
        text = f.read()

    sentences = nltk.sent_tokenize(text)
    current_tokens = []
    chunk_id = 0

    for sentence in sentences:
        sentence_tokens = tokenize(sentence)

        if len(current_tokens) + len(sentence_tokens) > CHUNK_SIZE:
            chunk_text = detokenize(current_tokens)
            all_chunks.append({
                "text": chunk_text,
                "source": txt_file.name,
                "chunk_id": chunk_id
            })
            chunk_id += 1
            current_tokens = current_tokens[-OVERLAP:]

        current_tokens.extend(sentence_tokens)

    if current_tokens:
        chunk_text = detokenize(current_tokens)
        all_chunks.append({
            "text": chunk_text,
            "source": txt_file.name,
            "chunk_id": chunk_id
        })

with open(OUT_FILE, "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, indent=2)

print(f"Saved {len(all_chunks)} chunks to {OUT_FILE}")
