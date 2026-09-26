# Phase 5 production-oriented backend

## Storage boundary

PostgreSQL 16 is canonical for mutable patient, review, finding, action, idempotency, rate-limit, and audit state. SQLAlchemy 2.x with psycopg 3 provides typed models and transactional repositories. Alembic owns schema changes; application startup never calls `create_all` or runs migrations in worker processes.

The version-controlled `source_registry.json`, `rule_evidence_registry.json`, processed source provenance, FAISS indexes and manifests, and evaluation fixtures remain governed files. This keeps evidence versions reproducible. FAISS remains the evidence index because its manifests and cosine semantics are already validated; a pgvector migration would add index-update and benchmark work without improving the current immutable, small corpus. Mutable operational state is never written to SQLite. Old local SQLite demo files are not automatically production-migrated.

```text
Client / clinician UI
         │
         ▼
FastAPI ── JWT/OIDC authentication ── RBAC ── request ID
         │
         ▼
Application services
  ├─ Patient/review/finding repositories ── PostgreSQL ── HMAC audit chain
  ├─ Offline deterministic clinical rules
  └─ Governed evidence service
      └─ lifecycle filter → FAISS cosine + cached BM25 → RRF
         → 30-candidate reranker → generator → verifier → grounded answer

JSON logs / Prometheus metrics / OpenTelemetry-compatible spans
```

## Schema and transactions

`patients` has a UUID key, unique source patient ID, current context hash, and current snapshot pointer. `patient_snapshots` stores immutable normalized JSONB with unique `(patient_id, context_hash)`. A reimport of unchanged semantics reuses the snapshot; a change creates a new one. `clinical_reviews` points to exactly one patient and snapshot, stores the review status and serialized review contract, and retains the original context hash. `clinical_findings` has immutable structured JSONB plus rule identity, status, context hash, and a clinical `DATE` for `evaluated_as_of`; `(review_id, rule_id, rule_version)` is unique. `finding_actions` is append-only and carries actor ID. Foreign keys use `RESTRICT`, so deleting a patient cannot silently erase reviews or findings. Database triggers reject UPDATE, DELETE, and TRUNCATE on snapshots, findings, actions, audit events, and checkpoints.

Each repository write opens an explicit transaction. Patient upsert uses a unique source ID and row lock. Review creation binds the snapshot in the same transaction. Review evaluation locks the review row before inserting findings, so concurrent requests cannot duplicate them. Action and clinical write transactions include their audit event. An audit failure rolls back the clinical write.

`idempotency_keys` stores actor subject, operation, key, SHA-256 request fingerprint, and logical response JSONB. A PostgreSQL transaction advisory lock serializes the same key across workers. The record is committed with the clinical write; same key and payload returns the original logical result, and changed payload conflicts. Keys currently do not expire. `rate_buckets` performs atomic per-minute upserts in PostgreSQL; it is shared by workers and contains no clinical payload.

## Migration workflow

The initial revision is `411c286b9ec8`. On an empty database, run `alembic upgrade head` before starting the API. Compose has a one-shot `migrate` service that waits for PostgreSQL health and completes before `api` starts. A downgrade is allowed only while all patient, audit, and idempotency tables are empty. It raises rather than silently dropping clinical history. Test the sequence on a disposable database: empty → upgrade → inspect tables and foreign keys → downgrade → upgrade. Back up production data and validate a migration on a copy before rollout.

## Authentication and authorization

Production `AUTH_MODE=oidc` verifies JWT signatures with cached JWKS keys, allows RS256 and ES256 only, and checks issuer, audience, expiration, issued-at, not-before when supplied, and subject. The JWKS fetch has a three-second timeout. Development mode is explicit and permits a local demo identity; its signed HS256 fixture requires a development-only key and validates issuer, audience, and time claims. There is no password database or hardcoded production token. Missing production OIDC configuration prevents startup.

| Role | Patient read | Import | Review/evaluate | Action | Evidence | Audit read | Governance mutation |
|---|---|---|---|---|---|---|---|
| clinician | Yes | Yes | Yes | Yes | Yes | No | No |
| clinical_admin | Yes | Yes | Yes | Yes | Yes | Yes | None exposed |
| guideline_editor | No | No | No | No | Yes | No | None exposed |
| auditor | No | No | No | No | No | Yes | No |

The API checks roles server-side before calling route handlers. Missing or invalid credentials return 401 in OIDC mode; authenticated actors with insufficient roles receive 403. The typed request actor reaches patient repositories, but this single-organization prototype has no patient-specific access grants or tenant model; deployments with multiple organizations need an object access policy before storing their data. `/health/live` and `/health/ready` are public. `/metrics` is public in local development and limited to auditor or clinical admin in production.

## Audit integrity and threat model

`audit_events` stores sequence, UTC timestamp, actor subject and roles, request ID, event and resource identities, minimized JSONB payload, predecessor hash, SHA-256 event hash, and HMAC-SHA256. Serialization is canonical JSON with sorted keys and fixed separators. A single `audit_head` row is locked with `SELECT FOR UPDATE` in every append transaction. This establishes one monotonically ordered predecessor chain under concurrency. Each hundredth event creates a keyed `audit_checkpoint` in the same transaction. `python -m scripts.verify_audit_chain` checks sequence continuity, predecessors, hashes, HMACs, checkpoints, and the head; corruption returns a nonzero exit code.

The chain detects accidental or unauthorized mutation when the attacker lacks the audit key. Database triggers prevent normal updates and deletes. A fully privileged attacker who controls both the database and HMAC key can rewrite history; checkpoints are stored in the same database and are **not** external anchoring. Audit payloads omit the full FHIR Bundle, normalized context, medical narrative, prompts, tokens, and API keys. Evidence query audit stores operation status and request correlation, not raw query text.

## API reliability and privacy

All responses receive a validated or generated `X-Request-ID`, content type protection, referrer policy, and a conservative content security policy. Errors use a stable `error` object with code, message, and request ID; legacy `detail` is retained for existing clients. FHIR bodies are capped at 2 MB, evidence queries at 2,000 characters, notes at 2,000 characters, and list limits at 100. The PostgreSQL rate limiter protects evidence queries, imports, and evaluations. Pool size, overflow, timeout, pre-ping, and recycle are configurable. Groq calls have bounded timeout and retry settings; validation, authorization, and clinical-rule failures are not retried.

Production JSON access logs contain route, method, status, latency, a stable truncated subject hash, and request ID. They omit bodies and tokens. Prometheus metrics include request counts and latency, database pool state, evidence result counts, retrieval stages, verification latency, provider outcomes, retries and latency, review evaluations, and finding status counts. OpenTelemetry API spans mark HTTP requests, database transactions, retrieval, reranking, generation, verification, and deterministic rule evaluation; no exporter is configured by default. Trace attributes use IDs, counts and rule IDs, not patient context or evidence text. Readiness checks PostgreSQL plus registry and index provenance without invoking an LLM.

## Performance and model lifecycle

BM25 indexes are cached in a bounded 16-partition LRU keyed by selected evidence identities and text digests. Lifecycle and metadata filters run before partition lookup, so blocked evidence cannot affect scores. Any changed evidence text or selected partition produces a new key. Embedding, reranker, and verifier model objects are retained per process after lazy loading. Warm CPU profiling showed the reranker dominated Phase 4 latency. Passage length was bounded to 1,200 characters, model max length to 256 tokens, and 30 candidates are inferred in one batch. Development document metrics did not justify reducing the established 30-candidate pool, so it was retained. Subsequent held-out evidence-unit checks confirmed that choice. See `eval/final_phase5_report.json` for exact quality and latency.

Each worker loads its own models; more workers multiply memory use and do not provide free throughput. Lazy model loading and inference use per-process locks to prevent duplicate loads and unsafe concurrent calls on a shared model. The supplied Compose API uses one worker. Model weights may need to be downloaded on the first model-backed query. The deterministic smoke replaces provider and semantic models with controlled fakes; it never requires a Groq key.

## Configuration and deployment

Required production variables are `APP_ENV=production`, `DATABASE_URL`, `AUTH_MODE=oidc`, `OIDC_ISSUER`, `OIDC_AUDIENCE`, `OIDC_JWKS_URL`, `AUDIT_HMAC_KEY`, and explicit `CORS_ORIGINS`. `GROQ_API_KEY` is needed only for external generation. Optional pool, rate-limit, model, timeout, and log settings have typed defaults. `.env.example` has development placeholders only. Secrets must be injected at deployment; none are baked into the image. The Dockerfile pins Python 3.11.15, uses a non-root runtime user, and installs a CPU PyTorch wheel. Compose provides PostgreSQL 16.10 with a named volume and health check.

This is production-oriented infrastructure, not a claim of clinical validation or security compliance. Manual guideline governance, local single-node assumptions, model latency and memory, shared-database audit checkpoints, development identity limitations, and the absence of an automated CI gate remain known limitations.

The Phase 5 dependency audit found eight advisories in the resolved `transformers` 4.57.6 dependency. Published fixes require a 5.x major upgrade, which was not applied without sentence-transformers compatibility and retrieval regression testing. `nltk`, used by offline corpus-building scripts, was removed from runtime requirements and remains in development requirements. Deployments should re-audit their exact locked dependency set and restrict model sources until this upstream migration is validated.
