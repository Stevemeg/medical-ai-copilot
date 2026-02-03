import subprocess
from embeddings.retrieve import retrieve

SYSTEM_PROMPT = """
You are a medical knowledge assistant.

Use the provided context to answer the question.
Do NOT use outside knowledge.
If the context does not contain the answer, clearly say:
"I don't know based on the provided medical documents."

Write a clear, well-structured medical explanation.
"""

def sanitize_text(text: str) -> str:
    return (
        text
        .replace("\uf0b7", "-")
        .replace("", "-")
        .replace("", "-")
        .encode("utf-8", errors="ignore")
        .decode("utf-8")
    )


def call_llm(prompt: str) -> str:
    result = subprocess.run(
        ["ollama", "run", "mistral"],
        input=prompt.encode("utf-8"),   #  BYTES, NOT STRING
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    return result.stdout.decode("utf-8", errors="ignore").strip()



def answer_question(question: str, top_k: int = 5):
    retrieved_chunks = retrieve(question, top_k=top_k)

    context = "\n\n".join(
        f"Source: {c['source']}\n{c['text']}"
        for c in retrieved_chunks
    )

    full_prompt = f"""
{SYSTEM_PROMPT}

Context:
{context}

Question:
{question}

Answer:
"""


    answer = call_llm(full_prompt)

    sources = list({c["source"] for c in retrieved_chunks})

    return {
        "answer": answer,
        "sources": sources
    }

if __name__ == "__main__":
    while True:
        q = input("\nAsk a medical question (or 'exit'): ")
        if q.lower() == "exit":
            break

        result = answer_question(q)
        print("\nAnswer:\n", result["answer"])
        print("\nSources:")
        for s in result["sources"]:
            print("-", s)
