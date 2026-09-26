# Medical AI Copilot

A production-oriented clinical review and governed evidence prototype with PostgreSQL state, authenticated role-based APIs, tamper-evident audit, deterministic clinical rules, grounded evidence queries, and observable deployment infrastructure.

The bundled patients are synthetic. Findings support clinician review; the application does not diagnose, prescribe, or select medication. It is not clinically validated or certified for regulated use.

## What it does

- Ingests a scoped FHIR R4-compatible synthetic Bundle into normalized patient context.
- Stores immutable context snapshots and binds each clinical review to one snapshot.
- Evaluates deterministic, evidence-bound rules without an LLM and records immutable findings.
- Records clinician dispositions as separate append-only actions.
- Retrieves governed evidence with lifecycle filtering before cosine and BM25 ranking, then reranks candidates and verifies generated claims. Unsupported claims are excluded.
- Exposes `/v1/` APIs with JWT/OIDC verification, server-side roles, request IDs, bounded inputs, PostgreSQL rate limiting, and keyed audit integrity.

## Architecture

```text
Client / clinician UI
         │
         ▼
  FastAPI service ── authentication ── RBAC ── request context
         │
         ▼
  Application services
   ├─ patient and review repositories ── PostgreSQL ── durable audit
   ├─ deterministic clinical rules
   └─ governed evidence service
       └─ policy filter → FAISS cosine + cached BM25 → RRF
          → bounded reranker → generator → independent verifier

  Structured logs / Prometheus metrics / OpenTelemetry-compatible spans
```

Version-controlled source and rule evidence registries, processed artifacts, and FAISS manifests remain authoritative for evidence governance. PostgreSQL holds mutable application state. See [production architecture](PRODUCTION_ARCHITECTURE.md) for schema, security, operations, and limitations.

## Local deployment

Requires Docker Compose. The Compose file uses a PostgreSQL 16 container and an explicit migration service. Its default credentials are development-only examples.

```bash
cp .env.example .env
# Replace the development JWT and audit keys in .env.
docker compose up -d --build
docker compose ps
docker compose exec api python -m scripts.docker_smoke
docker compose exec api python -m scripts.verify_audit_chain
```

The PostgreSQL host port is `55432`; the API is at `http://localhost:18000`. `/health/live` checks process liveness; `/health/ready` checks PostgreSQL and governed evidence provenance. `/metrics` serves Prometheus text. The explicit migration service completes before the API starts; do not run migrations independently in every worker.

For local Python development, set `APP_ENV`, `DATABASE_URL`, `AUTH_MODE`, and `AUDIT_HMAC_KEY` as shown in `.env.example`, then run `alembic upgrade head` and `uvicorn api_server:app`. Production requires `AUTH_MODE=oidc` with issuer, audience, and JWKS URL. It rejects missing critical configuration and wildcard CORS.

The old SQLite files are local demo state and are **not** automatically migrated to PostgreSQL. The initial Alembic migration creates a clean production schema; historical clinical data is never silently erased by downgrade.

## API and authentication

The stable API prefix is `/v1`. Clinical routes require `clinician` or `clinical_admin`. Evidence query also accepts `guideline_editor`. Audit read requires `auditor` or `clinical_admin`. The compatibility `/api/ask` route uses the same governed evidence engine. In explicit development mode, a local anonymous clinician identity is available for the demo; signed short-lived development JWTs are also supported. Production only accepts verified OIDC JWTs.

Write routes accept `Idempotency-Key`. PostgreSQL stores the actor, operation, request fingerprint, and logical response in the same transaction as the write. Reuse with a different request returns a conflict. List routes have bounded limits. Audit events contain identifiers, hashes, counts, statuses, and correlation IDs, not raw FHIR Bundles or medical narrative.

## Verification

```bash
python -m pytest -q
PHASE5_TEST_DATABASE_URL=postgresql+psycopg://medical:development_only_change_me@localhost:55432/medical_test python -m pytest -q tests/test_phase5_postgres.py
python -m eval.final_phase5_benchmark
ruff check .
ruff format --check .
mypy backend/settings.py backend/db.py backend/security.py backend/durable_audit.py backend/postgres_store.py backend/operations.py
```

The benchmark requires locally available embedding and reranker models but no Groq request. Unit tests use fakes; PostgreSQL integration tests require a separately migrated test database. The Docker smoke uses a deterministic fake generator and verifier while exercising the real API, retrieval filters, PostgreSQL workflows, and audit chain.

## Further reading

- [Knowledge governance](KNOWLEDGE_GOVERNANCE.md)
- [Patient context](PATIENT_CONTEXT.md)
- [Clinical rules](CLINICAL_RULES.md)
- [Phase 4 evidence design](EVIDENCE_PHASE4.md)
- [Production architecture and operations](PRODUCTION_ARCHITECTURE.md)
- [Compliance considerations](COMPLIANCE_CONSIDERATIONS.md)

No real patient information is included. Manual source lifecycle governance remains required. Models and endpoints need independent clinical, privacy, and deployment review before any clinical use.
