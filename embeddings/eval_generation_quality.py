"""
Evaluation harness -- NOT part of the app. Runs a small set of real
questions through the full answer_question() pipeline and prints the raw
output, so we can observe generation-quality behavior (does the model
hedge, contradict itself, or answer cleanly?) both BEFORE and AFTER any
system-prompt change, with a real before/after comparison rather than
just trusting that a prompt rewrite helped.

Usage:
    python embeddings/eval_generation_quality.py

Requires GROQ_API_KEY to be set in .streamlit/secrets.toml, since this
calls the real LLM through the same answer_question() function the app
uses.
"""

import sys
from pathlib import Path

# This script lives in embeddings/, so the project root is one level up.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.rag_pipeline import answer_question

# Three categories, deliberately chosen using questions already confirmed
# to route correctly (so any issue observed here is a GENERATION problem,
# not a retrieval problem -- that's already been separately verified).

CLEARLY_ANSWERABLE = [
    "How should a diabetic foot ulcer be managed?",
    "What are the risk factors for hypertension?",
    "When should statins be offered for cardiovascular risk reduction?",
]

BORDERLINE = [
    # The original failure case: context has the underlying mechanism
    # (cell types, receptor activation) but not the question's exact
    # phrasing assembled as one clean answer.
    "Explain insulin resistance.",
    "What is the SINBAD classification?",
]

GENUINELY_UNANSWERABLE = [
    # These should retrieve NOTHING (confirmed by the relevance gate
    # working correctly in earlier testing) and never reach the LLM at
    # all -- included here as a sanity check that the short-circuit still
    # works, not because we expect a generation-quality issue here.
    "How do I bake a chocolate cake?",
    "What's the capital of France?",
]


def run_category(label, questions):
    print(f"\n{'=' * 70}")
    print(label)
    print(f"{'=' * 70}")
    for q in questions:
        print(f"\n--- Q: {q!r} ---")
        result = answer_question(q)
        print(f"Answer:\n{result['answer']}")
        print(f"\nSources: {result['sources']}")
        print("-" * 70)


run_category("CLEARLY ANSWERABLE", CLEARLY_ANSWERABLE)
run_category("BORDERLINE (the original insulin-resistance failure mode)", BORDERLINE)
run_category("GENUINELY UNANSWERABLE (should short-circuit, no LLM call)", GENUINELY_UNANSWERABLE)