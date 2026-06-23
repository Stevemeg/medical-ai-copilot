from embeddings.retrieve import retrieve
from groq import Groq
import streamlit as st

from backend.config import get_groq_api_key
from backend.audit_log import log_interaction

# -----------------------------------
# Groq Client
# -----------------------------------

def get_groq_client():
    # get_groq_api_key() checks environment variables first (the real
    # production path, matching how cloud secrets managers actually
    # deliver secrets), falling back to .streamlit/secrets.toml for local
    # development. See backend/config.py for the full explanation.
    return Groq(
        api_key=get_groq_api_key()
    )

# -----------------------------------
# System Prompt
# -----------------------------------

SYSTEM_PROMPT = """
You are a medical knowledge assistant. Answer using ONLY the provided
context -- never use outside knowledge.

Before writing your answer, read ALL of the provided context chunks
together as a whole, not one at a time. Medical context is often spread
across multiple chunks that each describe a different piece of the SAME
mechanism or recommendation -- combine these freely, even if no single
chunk states the answer in full.

Then commit to exactly ONE of these two modes -- never blend them. These
mode labels are for your own internal decision-making only -- never write
the words "MODE 1" or "MODE 2" in your actual answer.

MODE 1 (sufficient context): Write a direct, clear, well-structured
answer. State the answer plainly. Do not hedge, do not say the context
"doesn't explicitly" cover something if the combined context still
supports a confident answer, and do not mention what the context lacks.

MODE 2 (insufficient context): If, after considering all chunks together,
they genuinely do not support an answer, write exactly: "I don't know
based on the provided medical documents." Do not add a partial answer
before or after this sentence.

Never do both -- never hedge through several paragraphs and then state a
confident answer anyway, and never give a confident answer and then
undercut it by saying you don't know. Pick one mode and commit to it.
Begin your answer directly with the substantive content -- do not open
with a label, a restatement of these instructions, or a meta-comment
about which mode you chose.

This applies to your closing sentence just as much as your opening one.
If you've written a confident MODE 1 answer, end it there -- do not add a
final caveat noting that the context "doesn't explicitly discuss" the
topic "as a separate entity," doesn't mention something "directly," or
similar. That closing-sentence hedge is the same error as opening with
one; once you've committed to MODE 1, nothing later in the answer should
walk it back.
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


def format_page_range(page_start, page_end) -> str:
    """
    Formats a chunk's page range for citation. Returns an empty string if
    page info isn't available (e.g. older chunk data from before page
    tracking was added), so citations degrade gracefully rather than
    showing "page None".
    """
    if page_start is None or page_end is None:
        return ""
    if page_start == page_end:
        return f", page {page_start}"
    return f", pages {page_start}-{page_end}"

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
        max_tokens=1024,
        # Defensive measure against repetition-loop degeneration -- found
        # in testing: a complex prompt caused the model to repeat the same
        # two sentences ~15 times until max_tokens cut the response off
        # mid-sentence. frequency_penalty discourages exact-phrase
        # repetition directly, independent of prompt wording. Groq's
        # documentation on support for this parameter is inconsistent
        # across sources -- verify this is actually being applied (e.g. by
        # checking whether a repetition loop still occurs) rather than
        # trusting it silently works.
        frequency_penalty=0.3
    )

    return response.choices[0].message.content.strip()

# -----------------------------------
# Main RAG Pipeline
# -----------------------------------

def answer_question(question: str, top_k: int = 5):

    retrieved_chunks = retrieve(question, top_k=top_k)

    if not retrieved_chunks:
        no_info_answer = "I don't have relevant information on this in the indexed medical documents."
        log_interaction(question, no_info_answer, [])
        return {
            "answer": no_info_answer,
            "sources": []
        }

    context = "\n\n".join(
        f"Source: {c['source']}{format_page_range(c.get('page_start'), c.get('page_end'))}\n"
        f"{sanitize_text(c['text'])}"
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

    # Deduplicate by (source, page range) rather than just source, so two
    # different page ranges from the same document show up as distinct,
    # individually citable entries instead of collapsing into one vague
    # filename-only reference.
    seen = set()
    sources = []
    for c in retrieved_chunks:
        page_range = format_page_range(c.get("page_start"), c.get("page_end"))
        label = f"{c['source']}{page_range}"
        if label not in seen:
            seen.add(label)
            sources.append(label)

    log_interaction(question, answer, sources)

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