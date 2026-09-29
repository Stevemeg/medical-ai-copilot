# Guideline intelligence and controlled activation

## Authority and trust

The version-controlled source registry, rule evidence registry, chunks and FAISS manifests remain immutable release inputs. PostgreSQL adds a governed structured-release layer; it does not mutate the old PDFs or relabel NG19/NG28. Manually importing recommendations requires an authenticated guideline_editor or clinical_admin and a registered clinical guideline's official HTTPS origin. A clinician cannot prepare, approve, reject or activate releases.

A manually supplied official URL is an attribution, not proof that entered text came from that page. The reviewer must compare each imported recommendation against authoritative text and document redistribution permission in `license_note`. Use synthetic content only in isolated demonstrations. There is no generic arbitrary-URL downloader, automatic crawler, or facility to turn a downloaded PDF into current evidence.

`python -m scripts.check_guideline_updates` reads `data/guideline_monitoring.json`. Current NICE sources are explicitly manual because this project has no verified stable structured recommendation feed. Registered structured-feed adapters can report version/date/checksum changes. The command returns nonzero on acquisition/validation failure and never activates content. Manual candidate registration is the supported path for the bundled sources.

## Acquisition and parser boundary

Acquisition requires HTTPS, exact registered origin, public DNS addresses, TLS hostname validation, size limit (1 MB for guideline JSON), expected JSON MIME, and consistent Content-Length. The transport pins a validated DNS address, rejects private/loopback/link-local IPv4 and IPv6, rejects user-info and unexpected ports/schemes, refuses all redirects, rejects compression, and caps response reads. Socket timeouts bound stalled I/O. Production egress policy should supplement application SSRF protection.

Only bounded JSON recommendation releases are parsed: at most 300 recommendations, 2500 characters each, bounded IDs/sections/pages and a real date. PDF/archive parsing is deliberately outside automatic acquisition, so no embedded PDF programs, macro execution, decompression or unbounded document parser runs. The existing offline corpus-ingestion tools remain separate.

## Workflow

```text
registered structured candidate
  -> quarantined (canonical JSON SHA-256; not retrievable)
  -> prepare: validate checksum, extract typed recommendations, textual diff,
              pinned normalized embeddings
  -> review_required
  -> approved -> activated
  -> rejected (terminal)
```

Preparation failure leaves the candidate quarantined for inspection; it never partially publishes vectors. Rejection follows review_required. Ambiguous matches block approval and require a corrected new candidate. Approval and activation are separate explicit API actions with meaningful notes. Reusing an existing version ID is forbidden.

- `GET /v1/guidelines`, `GET /v1/guidelines/{document_id}/versions`
- `GET /v1/guideline-updates` includes checksums, states and full structured diffs
- `POST /v1/guideline-updates` registers the `CandidateInput` schema in `/docs`
- `POST /v1/guideline-updates/{id}/prepare`
- `POST /v1/guideline-updates/{id}/approve`, `/reject`, `/activate`, with `{"note":"reason for this decision"}`

The UI shows documents, current/historical versions, candidate state/checksum, change counts, changed old/new text, and affected rules. Controls appear only for authenticated editor/admin roles; the API independently enforces those permissions.

## Diff and clinical impact

Stable recommendation IDs match exactly. Normalized whitespace-only edits are unchanged; matched-ID text changes have mandatory deterministic unified diffs. Unmatched stable IDs are added/removed. Without IDs, identical text in a unique section can remain unchanged, while changed/duplicate/ambiguous sections remain ambiguous. The source texts, rather than an LLM interpretation of a medical change, are authoritative. Metadata-only changes are recorded separately when recommendation text is unchanged.

Changed/removed rule-linked recommendation IDs surface rule ID/version and evidence reference. Activation conservatively suppresses every rule dependent on the replaced document, including metadata-only replacements, until a versioned rule/evidence review incorporates the release. No rule is automatically rewritten or allowed to claim the new text is verified-current. Existing immutable findings retain the evidence and patient snapshot used at their original evaluation.

## Atomicity and historical preservation

Each activation locks its candidate and takes a transaction-scoped per-document PostgreSQL advisory lock, then checks its recorded base against the active version. Concurrent candidates based on an obsolete release cannot both activate. The active pointer and critical HMAC audit event commit together. A failed audit rolls back activation. The pointer yields one current release; previous baseline/activated releases are superseded and remain available to explicit historical retrieval. Retrieval reads complete source/vector payloads under one database statement/snapshot and checks hashes before scoring. Database triggers protect approved/rejected/activated payloads and prohibit history deletion/truncation.

The source registry and generated baseline files do not change during online activation. To make a portable offline release, export/review a new version-controlled corpus through the existing artifact pipeline and reconcile mutable activation state under a separate reviewed migration. There is no automatic destructive reset or export command.

The isolated PostgreSQL tests demonstrate baseline v1 -> quarantined v2 -> review -> approval -> activation, actual current/historical retrieval, dependent-rule suppression, authorization rejection, concurrent obsolete-base rejection, audit rollback and chain verification. No current NICE PDF was fabricated or redistributed in this phase.

For a local Compose editor demonstration only, generate a short-lived token with `docker compose exec -T api python -c "from scripts.docker_smoke import token; print(token('guideline_editor'))"`, then paste it into Session access. Treat the output as a credential; do not save it in source, screenshots or logs. Production must obtain tokens from its configured OIDC provider.
