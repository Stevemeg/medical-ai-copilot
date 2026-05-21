from embeddings.retrieve import retrieve
from groq import Groq
import streamlit as st

# -----------------------------------
# Groq Client
# -----------------------------------

def get_groq_client():
    return Groq(
        api_key=st.secrets["GROQ_API_KEY"]
    )

# -----------------------------------
# System Prompt
# -----------------------------------

SYSTEM_PROMPT = """
You are a medical knowledge assistant.

Use ONLY the provided context to answer the question.

Do NOT use outside knowledge.

If the context does not contain the answer, clearly say:
"I don't know based on the provided medical documents."

Write a clear, accurate, and well-structured medical explanation.
"""

# -----------------------------------
# Text Cleaning
# -----------------------------------

def sanitize_text(text: str) -> str:
    return (
        text
        .replace("\uf0b7", "-")
        .replace("", "-")
        .replace("", "-")
        .encode("utf-8", errors="ignore")
        .decode("utf-8")
    )

# -----------------------------------
# LLM Call
# -----------------------------------

def call_llm(prompt: str) -> str:

    client = get_groq_client()

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2,
        max_tokens=1024
    )

    return response.choices[0].message.content.strip()

# -----------------------------------
# Main RAG Pipeline
# -----------------------------------

def answer_question(question: str, top_k: int = 5):

    retrieved_chunks = retrieve(question, top_k=top_k)

    context = "\n\n".join(
        f"Source: {c['source']}\n{sanitize_text(c['text'])}"
        for c in retrieved_chunks
    )

    full_prompt = f"""
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

# -----------------------------------
# CLI Testing
# -----------------------------------

if __name__ == "__main__":

    while True:

        q = input("\nAsk a medical question (or 'exit'): ")

        if q.lower() == "exit":
            break

        result = answer_question(q)

        print("\nAnswer:\n")
        print(result["answer"])

        print("\nSources:")
        for s in result["sources"]:
            print("-", s)