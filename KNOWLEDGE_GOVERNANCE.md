# Knowledge governance (Phase 1)

This repository is an educational prototype. Its default retrieval scope is registered, ingested **current clinical guidelines** only. It does not determine what is clinically appropriate for a patient.

## Source lifecycle

Authoritative source → `data/source_registry.json` → document/version validation → exact PDF SHA-256 → page parser → evidence chunks → FAISS index and manifest → lifecycle-aware retrieval.

The registry is the source of truth for canonical document identity, version identity, publisher, source type, jurisdiction, dates, local path, canonical URL, lifecycle state and checksum. Unknown dates remain `null`. `KnowledgeDocument` identifies a work independent of filename; `KnowledgeDocumentVersion` identifies the exact local snapshot. `EvidenceChunk` identifies a text span and its page range. Chunk IDs are stable for unchanged versions and chunking configuration. Section and recommendation IDs remain unknown (`null`) because the current PDF parser does not reliably extract them.

Locally stored sources are verified on registry load. A missing file or changed SHA-256 stops startup or ingestion. Processed-page sidecars bind extracted JSON to the exact source hash and parser version; unchanged validated sources skip extraction. A chunk manifest binds chunks to the exact processed files and chunker configuration. FAISS manifests bind each index to its metadata hash, registry hash, model name, chunker configuration and source versions/hashes. Missing or stale manifests stop retrieval. A source or registry change requires re-extraction/re-chunking/re-indexing; old artifacts cannot silently bypass the new policies. The committed legacy vectors were migrated only after checking vector counts and exact chunk/metadata alignment. Their embedding library revision at original creation is unverified; future rebuilds use the documented model name.

## Evidence classes and policies

| Policy | Eligible evidence |
|---|---|
| `current_clinical` (default) | Ingested `clinical_guideline` versions marked `current` |
| `historical_only` | Ingested `superseded` or `historical` versions, requested explicitly |
| `historical_allowed` | Ingested current, superseded and historical versions, requested explicitly |
| `reference` | Ingested textbook or patient education sources, requested explicitly; withdrawn material excluded |

`unknown` and `withdrawn` are excluded from clinical and historical modes. Public health reports and original epidemiological research describe population data and never enter the default treatment-guideline search. Textbooks and patient education are reserved for explanatory use. The retrieval code constructs a filtered FAISS view **before** searching or applying the relevance gate, so blocked vectors cannot influence ranking. BM25/RRF run on the same filtered evidence. Historical and reference retrieval are internal/API options; the Streamlit UI uses the clinical default.

`POST /api/ask` accepts optional `policy` using the values above. The response preserves legacy `sources` labels for existing UI rendering and adds structured `citations` with chunk/document/version IDs, title, publisher, dates, source type, jurisdiction, lifecycle, URL, and page range. These are retrieved-context citations, not claim-level verification. The API does not decide which policy is appropriate for a patient.

## Corpus audit (verified 24 September 2026)

| Bundled source | Publisher / type / jurisdiction | Snapshot | Lifecycle | Action |
|---|---|---|---|---|
| NICE NG19 diabetic foot | NICE / guideline / UK | Updated 11 Oct 2019 | Current | Included in default clinical retrieval; [NICE listing](https://www.nice.org.uk/guidance/ng19) matches PDF |
| NICE NG136 hypertension | NICE / guideline / UK | Updated 26 Feb 2026 | Current | Included; [NICE listing](https://www.nice.org.uk/guidance/ng136) matches PDF |
| NICE NG238 cardiovascular risk | NICE / guideline / UK | Published 14 Dec 2023; no later date in PDF | Current | Included; [NICE guidance](https://www.nice.org.uk/guidance/ng238) identifies this edition |
| NICE NG28 type 2 diabetes | NICE / guideline / UK | Updated 29 Jun 2022 | Superseded | Retained for explicit historical use; [NICE now lists 18 Feb 2026](https://www.nice.org.uk/guidance/ng28) |
| Ministry of Health and Family Welfare diabetes FAQ | India MoHFW / patient education / India | Date unverified | Unknown | Excluded from clinical guidance; [official PDF](https://ncdc.mohfw.gov.in/NCDC_MOHFW/uploads/pdf/Diabetes%20Mellitus%20English%20FAQs.pdf) |
| WHO global TB report | WHO / public health report / WHO-global | 2020 | Historical | Retained for historical retrieval; [WHO 2020 record](https://www.who.int/publications/i/item/9789240013131), [2025 report](https://www.who.int/teams/global-programme-on-tuberculosis-and-lung-health/tb-reports/global-tuberculosis-report-2025/) |
| WHO world malaria report | WHO / public health report / WHO-global | 2019 | Historical | Retained for historical retrieval; [WHO 2019 record](https://www.who.int/publications/i/item/9789241565721), [2025 report](https://www.who.int/publications/i/item/9789240117822) |
| CDC multiple chronic conditions article | CDC / original research / US | Published 17 Apr 2025; data through 2023 | Historical snapshot | Excluded from guideline retrieval; [CDC article](https://www.cdc.gov/pcd/issues/2025/24_0539.htm) |
| OpenStax Anatomy & Physiology | OpenStax / textbook / global | 2013 edition indicated in PDF; exact revision unverified | Unknown | Explicit reference only; [OpenStax source](https://openstax.org/books/anatomy-and-physiology/pages/1-introduction) |

The older NICE PDF is not replaced with a downloaded 2026 PDF because its redistribution rights were not established. The same caution applies to new publisher material. Existing PDFs remain in the repository; their rights should be reviewed before any redistribution or commercial deployment. No snapshot is marked current solely because its filename sounds recent. Currentness is a registry assertion last checked on the audit date, not a live publisher feed.

## Rebuild and validation

Requires Python 3.11+. Install `requirements.txt` (or `requirements-dev.txt` for checks). From the repository root:

```bash
python -m embeddings.extract_text
python -m embeddings.chunk_text
python -m embeddings.build_faiss_index
pytest
```

Before registering a new source, verify the publisher, type, jurisdiction, lifecycle and redistribution rights; create a stable document ID and version ID; record the exact local SHA-256 and known dates/URL. Update the registry first. The three rebuild commands then validate raw bytes, processed provenance and index manifests. Unchanged validated PDFs skip extraction, and an index with exactly matching metadata and manifest skips re-embedding. If the PDF content changes, register a new version and rebuild; do not edit the hash to make an old version appear unchanged. The current PDF parser can be memory-intensive for the large textbook, and the chunker requires NLTK punkt data. `nltk.download` runs in the chunking script if the tokenizer data are absent.

The one-time `embeddings.migrate_legacy_artifacts` command was used only for the committed pre-Phase-1 snapshot; use the normal rebuild path for later changes. A committed index manifest is checked at application startup, including when an old index is present. The ungoverned mixed index and its metadata were removed from the repository.

## Limits

Publisher pages may change after this audit, so currentness needs periodic human review. Source dates absent from the source or publisher remain unknown. The original source-retrieval dates are unknown. Page ranges can be imprecise near overlapping chunk boundaries. The system has no reliable section/recommendation extraction, no claim-level grounding, and no clinical validation. The existing LLM generation path remains a research/educational feature, and its answers require professional review.
