# Security and operational threat model

## Trust boundaries

FHIR/browser input, OAuth responses, remote guideline bytes and retrieved text are untrusted. Server-side Pydantic/normalization, provenance checks, exact-origin SSRF transport, JWT verification and role checks precede privileged operations. Clinical rules use deterministic code. Generated claims pass independent semantic and citation checks; failure/unavailability abstains. Retrieved instructions cannot change lifecycle metadata or authorization.

Production accepts OIDC RS256/ES256 with verified issuer/audience/expiry/signature and bounded JWKS retrieval. Development HS256 tokens require an explicit test key and the fixed development issuer/audience. Unverified JWT header inspection only chooses an allowed verification algorithm; payload authorization always follows signature verification. Clinical routes require clinician/admin, evidence additionally editor, guideline mutation authenticated editor/admin, audit auditor/admin. CORS rejects wildcard production origins.

PostgreSQL row locks, advisory locks, unique constraints and idempotency coordinate concurrent writes. Immutable snapshot/finding and append-only action/audit triggers prevent normal application mutation. A clinical write and its critical audit append share a transaction. The HMAC chain includes a serialized head, sequence, previous hash, event payload and checkpoint checks; tests deliberately corrupt test events and verify rejection. There is no external immutable checkpoint anchor. A DB administrator with the signing key could rewrite all history; least privilege and externally secured checkpoints remain deployment work.

Logs use route templates, request IDs, statuses and pseudonymous actor identifiers. FHIR bodies, question text, tokens and authorization codes are excluded. OAuth query strings must not be logged by the reverse proxy; Uvicorn access logs are disabled. The UI stores a supplied JWT in memory only. The generator is not automatically given patient records. Do not put real PHI in any input.

Guideline/FHIR transport requires exact HTTPS origins, globally routable resolved addresses, a pinned TLS socket, expected MIME and bounded content. Redirects, credentials, private/link-local addresses, compression, unsupported schemes and ports are rejected. No fetched program/model code is executed. See GUIDELINE_INTELLIGENCE.md for parser limits and INTEROPERABILITY.md for state/PKCE/token boundaries.

## Dependency remediation

An isolated Python3.11 environment was used to move **Transformers 4.57.6 -> 5.17.0** and **sentence-transformers 2.7.0 -> 5.7.0**, with Torch2.14.0. Runtime and developer direct dependencies are exact-pinned in requirements files; Docker/CI use those same pins (CPU Torch wheels in Linux). Model revisions are pinned in model_manifest.json and remote model code is disabled. Do not accept unreviewed model repositories or serialized checkpoints from requests.

The pre-upgrade audit reported eight Transformers entries, including duplicate advisory aliases. Unique issues and exercised-path review:

| Advisory | CVE | Functionality / project exposure | Resolution |
|---|---|---|---|
| PYSEC-2025-217 | CVE-2025-14929 | X-CLIP conversion untrusted deserialization; conversion not used | No longer reported with 5.17.0 |
| PYSEC-2026-2288 | CVE-2026-1839 | Trainer RNG checkpoint loading; no training/resume endpoint | Upgrade beyond 5.0 fix |
| PYSEC-2026-2289 | CVE-2026-4372 | Model config attention implementation code load; model loading is used, so fixed version and pinned repositories required | Upgrade beyond 5.3 fix |
| PYSEC-2026-2290 | CVE-2026-5241 | LightGlue remote-code override; LightGlue not used | Upgrade beyond 5.5 fix |
| PYSEC-2026-3929 | CVE-2026-9856 | Tokenizer/processor save path traversal; no untrusted save API exposed | Upgrade beyond 5.10 fix |

The final audit wrapper surfaces one contained **development-only** advisory: NLTK3.10.3 `PYSEC-2026-3740` (`CVE-2026-81726`, `GHSA-8mgp-746c-j5xp`), model persistence path sandbox bypass. The chunker uses sentence tokenization, not transition-parser/perceptron/maxent model save/load APIs or caller-selected paths. No fixed version was reported. NLTK is omitted from production requirements/Docker. The exact version/advisory exception in `security_exceptions.json` expires 2026-10-29 and fails closed after expiry. This is containment, not a claim of zero development advisories.

Run `python -m scripts.security_audit` for the raw pip-audit JSON and reviewed-exception summary. No blanket ignore is used. Update dependencies in an isolated environment, inspect exact advisory paths/fixes, run the complete regression/model evaluation/Compose smoke, and review changed model behavior before updating pins. Transitive resolution remains package-index dependent; the current strategy pins critical direct dependencies rather than claiming a hash-complete transitive lock.

Bandit annotations are limited to fixed local subprocess commands (B404/B603, no shell; release Git SHA validated) and assertions in the executable Docker test harness (B101, rejects optimized Python). Each annotation has an inline scope/reason. CI Gitleaks scans full checked-out history with built-in ephemeral GitHub authentication. Local Gitleaks also scans current tracked/untracked source changes using an explicit staging directory so model caches and virtual environments are excluded rather than accidentally scanned as source.

## Operational retention

`RATE_BUCKET_RETENTION_HOURS=24` retains rate buckets beyond the active minute; minimum 1h. `IDEMPOTENCY_RETENTION_DAYS=7` defines the supported retry window; minimum1d. Expired SMART handoffs/verifiers are ephemeral and eligible after their bounded expiry (default300s).

```bash
python -m scripts.cleanup_operational_state          # dry run; counts only
python -m scripts.cleanup_operational_state --execute
```

Only rate_buckets, idempotency_keys and expired smart_launches are touched. PostgreSQL transaction time determines strict cutoffs, and `FOR UPDATE SKIP LOCKED` skips rows in use. Current-minute rate accounting and retry-window idempotency remain intact. Patients, patient_snapshots, clinical_reviews, clinical_findings, finding_actions, audit_events, audit_heads, audit_checkpoints, guideline_candidates and guideline_activations are protected from this cleanup by an explicit table allowlist in code. Clinical/audit retention is a separate future policy, not implied by operational cleanup.

Schedule hourly with a local cron/Kubernetes maintenance job under the deployment's secret-management policy, e.g. `docker compose exec -T api python -m scripts.cleanup_operational_state --execute`. GitHub Actions must not connect to a production DB. Dry-run and execution tests seed recent/old operational rows and verify clinical audit integrity. After the retry window, a repeated key may create a new operation; clients must reconcile old results rather than assume indefinite deduplication.

## Release operation

Use the one-shot migration service before starting upgraded API instances. Never run migrations in every worker. Test upgrade/downgrade/re-upgrade only on an empty disposable DB: supported downgrades reject populated immutable history rather than deleting it. Take backups and test restoration before real operational upgrades. Never regenerate an audit key for an existing persisted database without a separately designed key-rotation procedure.

The first hosted CI run also identified runner-provided setuptools79.0.1 (`PYSEC-2026-3447`); runtime requirements now pin the already locally audited setuptools84.0.0. The scanner audits the complete installed distribution inventory with resolution disabled. For the exact official PyTorch `2.14.0+cpu` wheel, it explicitly checks upstream `2.14.0` advisories and records that mapping; no package is silently skipped. Unknown skipped packages fail. GitHub test keys are generated per job and masked before being written to its environment.
