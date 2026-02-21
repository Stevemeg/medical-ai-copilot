#  Medical AI Copilot — Retrieval-Augmented Generation (RAG)

> **A local, privacy-first RAG system that answers medical questions using trusted clinical guidelines — with source attribution and zero hallucination risk from ungrounded LLM knowledge.**

>  For **educational and research purposes only**. Not a substitute for professional medical advice.

---

##  What Problem Does This Solve?

LLMs can confidently generate incorrect medical information — a risk that's unacceptable in healthcare. Standard chatbots have no mechanism to cite sources or constrain answers to verified content.

This project solves that by building a **Retrieval-Augmented Generation (RAG) system** that:
- Only answers from **curated, trusted medical documents** (WHO, NICE, Ministry of Health guidelines, OpenStax textbooks)
- **Cites its sources** with every response
- Runs **entirely offline** using a local LLM — no cloud API, no data leaving your machine

---

##  What Was Built

An end-to-end RAG pipeline where a user asks a medical question, the system retrieves the most relevant passages from real clinical documents, and a local LLM generates a grounded answer — backed by sources, not hallucinations.

---

##  How It Works

```
Medical PDFs (WHO, NICE, OpenStax, MoH Guidelines)
        │
        ▼
┌──────────────────────┐
│  Text Extraction &   │  → PDFs parsed, cleaned, unicode-safe
│  Chunking            │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Sentence Embeddings │  → SentenceTransformers converts chunks to dense vectors
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  FAISS Vector Index  │  → Fast similarity search over all document chunks
└──────────┬───────────┘
           │
        User Query
           │
           ▼
┌──────────────────────┐
│  Semantic Retrieval  │  → Top-K most relevant chunks fetched
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Local LLM (Ollama)  │  → Generates answer using ONLY retrieved context
└──────────┬───────────┘
           │
           ▼
  Grounded Answer + Source Documents 
```

---

##  Tech Stack

| Component | Tool |
|-----------|------|
| Language | Python |
| Embeddings | SentenceTransformers |
| Vector Store | FAISS |
| LLM Inference | Ollama (fully local) |
| UI | Streamlit |
| Domain | Medical / Healthcare AI |

**Why local LLM?** Patient queries are sensitive. Running Ollama locally means no data is sent to external APIs — privacy by design.

---

##  Project Structure

```
medical-ai-copilot/
│
├── data/
│   ├── raw_docs/          # Original medical PDFs
│   ├── processed_docs/    # Extracted & cleaned text
│   └── vector_store/      # FAISS index & metadata
│
├── embeddings/
│   ├── chunk_text.py      # Split docs into retrievable segments
│   ├── embed_text.py      # Generate dense vector embeddings
│   ├── build_faiss_index.py  # Index embeddings for fast search
│   └── retrieve.py        # Query-time semantic retrieval
│
├── backend/
│   └── rag_pipeline.py    # Core RAG orchestration logic
│
├── ui/
│   └── app.py             # Streamlit chat interface
│
├── README.md
└── requirements.txt
```

---

##  Example Questions the System Can Answer

- *What is diabetes?*
- *What are the types of diabetes?*
- *What are the risk factors for type 2 diabetes?*
- *How is diabetes managed according to guidelines?*

All answers are grounded in the retrieved document chunks and include document-level source attribution.

---

##  Key Features

- **RAG architecture** — answers constrained to retrieved medical content, not LLM imagination
- **Source attribution** — every answer shows which document(s) it came from
- **Fully offline** — Ollama runs locally, no API keys or internet required
- **Unicode-safe processing** — handles medical terminology and special characters cleanly
- **Interactive UI** — Streamlit front-end for easy querying

---

##  Why This Matters (For Recruiters)

This project demonstrates:
- **RAG system design** — building a production-style retrieval pipeline from scratch (chunking → embedding → indexing → retrieval → generation)
- **Vector search** — hands-on experience with FAISS and semantic similarity
- **LLM integration** — working with local inference via Ollama, understanding context window management
- **Healthcare AI awareness** — understanding why hallucination is a critical failure mode in medical contexts and engineering around it
- **Privacy-first thinking** — choosing local LLM inference as a deliberate architectural decision

---

##  Known Limitations

- Local LLM inference is slower than cloud-based APIs
- Answers are scoped to the documents indexed — gaps in the document library mean gaps in answers
- Not designed for real-time clinical decision support

---

##  Future Roadmap

- [ ] FastAPI backend for production deployment
- [ ] Switchable LLM support (local + API-based, e.g., GPT-4, Claude)
- [ ] Hallucination detection metrics (faithfulness, answer relevance scoring)
- [ ] Expanded medical domain coverage

---

## Disclaimer

This system provides informational responses based on indexed medical documents. It **does not provide medical diagnosis or treatment advice**. Always consult a qualified healthcare professional.

---

##  Contact

**[Your Name]**  
[your.email@example.com] | [LinkedIn URL] | [GitHub URL]
