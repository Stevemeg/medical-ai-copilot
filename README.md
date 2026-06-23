# Medical AI Copilot — RAG-Based Clinical Assistant

> A Retrieval-Augmented Generation (RAG) system that answers clinical
> questions using indexed medical guidelines, with hybrid semantic +
> keyword retrieval, page-level source citations, and a tamper-evident
> audit trail.
>
> For educational and research purposes only. Not a substitute for
> professional medical advice.

---

## Live Demo

https://medical-ai-copilot-usov2kkptkqwcbpgzappudd.streamlit.app/

## GitHub Repository

https://github.com/Stevemeg/medical-ai-copilot

---

## Problem Statement

Large Language Models can hallucinate incorrect medical information when
generating responses from parametric memory alone. In healthcare
contexts, factual reliability and traceability are critical.

This project addresses that with a RAG pipeline that:

- Retrieves only from a fixed, indexed set of trusted clinical guidelines
- Grounds every answer in retrieved context, never outside knowledge
- Cites page-level sources for every answer
- Logs every interaction to a tamper-evident audit trail

---

## System Architecture

The corpus is split into two separate indexes rather than one combined
index, because mixing a large anatomy/physiology textbook with much
smaller clinical guideline documents caused the textbook to dominate
retrieval results for clinical questions during early development:

```
Medical Documents (NICE, WHO, MoH, OpenStax)
        │
        ▼
Text Extraction (page-tracked) & Chunking
        │
        ▼
Sentence Embeddings (MiniLM)
        │
        ├──────────────┬──────────────┐
        ▼                              ▼
  Clinical Index                 Anatomy Index
  (FAISS + BM25)                 (FAISS + BM25)
        │                              │
        └──────────────┬───────────────┘
                        ▼
      Index selection by raw FAISS distance
   (lower distance wins; gate excludes an index
      entirely if neither method finds a match)
                        │
                        ▼
   Within the selected index: FAISS + BM25 results
      fused via Reciprocal Rank Fusion (RRF)
                        │
                        ▼
       Groq-hosted Llama 3.1 Inference
                        │
                        ▼
   Grounded, cited response + tamper-evident
              audit log entry
```

**Why hybrid retrieval matters here**: pure semantic search struggles
with exact clinical terms — drug abbreviations (ACEi, ARB), guideline
codes (NG19, NG136), and named classification systems (SINBAD) often
score weakly in embedding space despite being an exact, unambiguous
match in the actual text. BM25 keyword search catches these; FAISS
catches conceptual/mechanism questions BM25 would miss (e.g. "explain
insulin resistance"). Fusing both, rather than relying on either alone,
was a real fix for a real, reproducible failure found during testing.

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python |
| Frontend | Streamlit (custom theme, no default styling) |
| Embeddings | SentenceTransformers (MiniLM) |
| Vector Store | FAISS (dual index: clinical + anatomy) |
| Keyword Search | BM25 (rank_bm25), fused via Reciprocal Rank Fusion |
| LLM Inference | Groq API (Llama 3.1 8B) |
| Audit Logging | SQLite, hash-chained for tamper-evidence |
| Secrets | Environment-variable-first, `.streamlit/secrets.toml` fallback |
| Domain | Healthcare AI |

---

## Project Structure

```
medical-ai-copilot/
│
├── app.py                       # Streamlit UI
├── requirements.txt
├── README.md
├── COMPLIANCE_CONSIDERATIONS.md # HIPAA/FDA CDS analysis (educational, not legal advice)
├── PROJECT_NOTES.md             # Running development log, including open issues
├── .gitignore
│
├── .streamlit/
│   └── config.toml              # Theme (secrets.toml is gitignored)
│
├── backend/
│   ├── rag_pipeline.py          # Prompting, generation, answer assembly
│   ├── config.py                # Secrets resolution (env var → secrets.toml)
│   └── audit_log.py             # Tamper-evident hash-chained audit log
│
├── embeddings/
│   ├── extract_text.py          # PDF → page-tracked JSON
│   ├── chunk_text.py            # JSON → token-bounded chunks with page ranges
│   ├── build_faiss_index.py     # Builds the dual (clinical/anatomy) FAISS indexes
│   └── retrieve.py              # Hybrid BM25 + FAISS retrieval, RRF fusion
│
└── data/
    ├── raw_docs/                # Source PDFs
    ├── processed/                # Page-tracked extracted text
    └── vector_store/
        ├── clinical_faiss.index / clinical_metadata.json
        └── anatomy_faiss.index / anatomy_metadata.json
```

---

## Key Features

- Dual-index retrieval (clinical guidelines vs. anatomy/physiology
  reference), selected per-query by FAISS distance
- Hybrid BM25 + semantic search within the selected index, fused via RRF
- Page-level source citations, grouped per document with combined page
  ranges (e.g. `NICE NG19 — Diabetic Foot Problems · pp.6-7, 13-15`)
- A relevance gate that returns a clean "I don't have relevant
  information" response — with no LLM call at all — for genuinely
  out-of-scope questions
- A generation prompt explicitly designed to avoid self-contradiction
  (answering confidently, then hedging or reversing itself), fixed after
  a real failure mode was found and reproduced during testing
- Tamper-evident audit logging: every interaction is hash-chained, so
  modifying or deleting a past log entry breaks the chain in a
  detectable way
- Environment-variable-first secrets resolution, matching how real cloud
  secrets managers (AWS/Azure/GCP) actually deliver credentials, with a
  documented migration path for each

---

## Example Questions

- How should a diabetic foot ulcer be managed?
- When should statins be offered for cardiovascular risk reduction?
- What is the SINBAD classification?
- Explain insulin resistance.

---

## Indexed Sources

NICE NG19 (Diabetic Foot Problems), NICE NG136 (Hypertension), NICE NG238
(Cardiovascular Risk & Lipids), NICE NG28 (Type 2 Diabetes), MoH Diabetes
Mellitus Guideline, WHO Tuberculosis Report, WHO Malaria Report, CDC
Chronic Disease Overview, OpenStax Anatomy & Physiology.

---

## Deployment

Deployed on Streamlit Community Cloud. The FAISS indexes are committed
to the repository rather than rebuilt at deploy time — this is a
deliberate choice for a demo deployment: rebuilding on every cold start
would require the raw source PDFs to also be present and would add real
startup latency, for no benefit in a context where the corpus doesn't
change at runtime.

**Secrets**: `GROQ_API_KEY` is set via Streamlit Community Cloud's
Secrets management (Advanced settings). `backend/config.py` checks
`os.environ` first, which is how Streamlit Cloud actually exposes
root-level secrets, before falling back to `st.secrets` — the same code
path works unchanged locally and when deployed.

---

## Skills Demonstrated

- Retrieval-Augmented Generation (RAG) with hybrid retrieval (semantic +
  keyword), not just vector search alone
- Real debugging of a multi-stage retrieval pipeline: diagnosing and
  fixing a genuine corpus-imbalance bug, a relevance-threshold
  miscalibration, and a Reciprocal-Rank-Fusion design flaw — each found
  through actual reproduction and measurement, not assumption
- LLM prompt engineering to fix a real generation-quality failure mode
  (self-contradicting answers), verified with a before/after eval set
- Tamper-evident audit logging (hash-chained, with the tamper-detection
  property actually tested by deliberately corrupting the log and
  confirming detection)
- Secrets management patterns matching real cloud deployment practices
- Honest compliance analysis (HIPAA/FDA CDS), including catching and
  correcting an outdated regulatory reference during the writing process
- Custom Streamlit theming and UI design (no default component styling)
- End-to-end deployment to Streamlit Community Cloud

---

## Run Locally

### Clone Repository

```
git clone https://github.com/Stevemeg/medical-ai-copilot.git
cd medical-ai-copilot
```

### Install Dependencies

```
pip install -r requirements.txt
```

### Add Secrets

Create `.streamlit/secrets.toml`:

```toml
GROQ_API_KEY = "your_api_key"
```

(Alternatively, set `GROQ_API_KEY` as a real environment variable —
`backend/config.py` checks that first.)

### Start Application

```
streamlit run app.py
```

---

## Known Limitations

- Responses are limited to the indexed documents — this is a deliberate
  design choice (grounding over completeness), not a gap to fill with
  unrestricted LLM knowledge
- Not intended for clinical diagnosis or treatment decisions; see
  `COMPLIANCE_CONSIDERATIONS.md` for an honest (non-legal) analysis of
  what real clinical use would actually require
- A known, currently-unresolved issue: certain queries can retrieve a
  chunk from a different (but topically adjacent) guideline, which the
  model sometimes tries to incorrectly cross-reference rather than
  ignoring — logged in detail in `PROJECT_NOTES.md`, including a fix
  that was attempted, found to cause a worse regression, and reverted
- The audit log's tamper-evidence is real and tested, but Streamlit
  Community Cloud's filesystem is ephemeral — the log resets on
  redeploys/restarts on this specific free hosting tier. The code is
  correct; this tier doesn't give it persistent storage
- Free-tier deployment may have cold-start latency on first load

---

## Future Improvements

- Persistent storage for the audit log (a small hosted database, rather
  than local SQLite, would survive Streamlit Cloud's ephemeral
  filesystem)
- A fix for the cross-guideline reconciliation issue above, at the
  retrieval/context-assembly layer rather than another prompt
  instruction (a prompt-level attempt already caused a regression and
  was reverted — documented in `PROJECT_NOTES.md`)
- A custom HTML/CSS/JS frontend with a separate API backend was actually
  built and verified working during development, for full pixel-level UI
  control beyond Streamlit's component model. It was deliberately not
  adopted as the primary interface — a two-process setup (API server +
  frontend, both needing to run and communicate correctly) was judged too
  much operational risk for a public demo link, against a Streamlit
  version that already worked well
- Expanded clinical guideline coverage
- PDF upload and dynamic re-indexing

---

## Disclaimer

This project provides informational responses based on indexed medical
documents and is intended for educational and research purposes only. It
does not provide medical diagnosis, treatment recommendations, or
professional healthcare advice. Always consult a qualified healthcare
professional for medical concerns.