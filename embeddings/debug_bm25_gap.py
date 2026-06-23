"""
Diagnostic script -- NOT part of the app. Measures BM25's rank-1 vs rank-2
score GAP (not the absolute score) for both irrelevant and exact-term
queries, since calibrate_threshold.py already showed absolute BM25 scores
don't cleanly separate these groups on this corpus (common-word noise in
multi-word natural-language queries inflates irrelevant scores enough to
overlap with genuine exact-term matches).

The hypothesis: a genuine exact-term match (like "SINBAD") should show a
LARGE gap between the best-matching chunk and the next-best, because only
one chunk actually contains the term. An irrelevant query's "best match"
is likely incidental word overlap that doesn't concentrate as strongly in
any single chunk, so the gap should be smaller.

Usage:
    python embeddings/debug_bm25_gap.py
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import retrieve

IRRELEVANT_QUESTIONS = [
    "What's the best programming language for web development?",
    "Who won the last World Cup?",
    "How do I bake a chocolate cake?",
    "What's the capital of France?",
    "Explain how a car engine works.",
    "What is the stock price of Apple today?",
    "How do I fix a leaking faucet?",
    "What's a good recipe for pasta carbonara?",
]

EXACT_TERM_QUESTIONS = [
    "What is the SINBAD classification?",
    "ACEi vs ARB for hypertension",
    "What is NG136?",
    "What is NG19?",
]


def measure_gap(label, questions, bm25_index, metadata):
    print(f"\n{'=' * 70}")
    print(label)
    print(f"{'=' * 70}")
    gaps = []
    gap_ratios = []
    for q in questions:
        scores = bm25_index.get_scores(retrieve.simple_tokenize(q))
        ranked = np.argsort(scores)[::-1]
        top1_score = scores[ranked[0]]
        top2_score = scores[ranked[1]]
        gap = top1_score - top2_score
        gap_ratio = (top1_score / top2_score) if top2_score > 0 else float("inf")
        gaps.append(gap)
        gap_ratios.append(gap_ratio if gap_ratio != float("inf") else top1_score)
        top1_source = metadata[ranked[0]].get("source", "?")
        print(f"  {q!r:55s}")
        print(f"    rank1={top1_score:7.3f}  rank2={top2_score:7.3f}  gap={gap:7.3f}  ratio={gap_ratio:.2f}  -> {top1_source}")
    gaps = np.array(gaps)
    print(f"\n  Gap (absolute) stats: min={gaps.min():.3f} max={gaps.max():.3f} mean={gaps.mean():.3f}")
    return gaps


print("Testing CLINICAL index")
clinical_irrelevant_gaps = measure_gap("CLINICAL -- irrelevant questions", IRRELEVANT_QUESTIONS, retrieve.clinical_bm25, retrieve.clinical_metadata)
clinical_exact_gaps = measure_gap("CLINICAL -- exact-term questions", EXACT_TERM_QUESTIONS, retrieve.clinical_bm25, retrieve.clinical_metadata)

print(f"\n{'=' * 70}")
print("CLINICAL GAP COMPARISON")
print(f"{'=' * 70}")
print(f"Irrelevant gap range: {clinical_irrelevant_gaps.min():.3f} to {clinical_irrelevant_gaps.max():.3f}")
print(f"Exact-term gap range: {clinical_exact_gaps.min():.3f} to {clinical_exact_gaps.max():.3f}")
if clinical_exact_gaps.min() > clinical_irrelevant_gaps.max():
    suggested = (clinical_exact_gaps.min() + clinical_irrelevant_gaps.max()) / 2
    print(f"\nClean separation by GAP found. Suggested gap floor: {suggested:.2f}")
else:
    print("\nStill overlapping by gap. We may need a different signal entirely.")