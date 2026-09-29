# Interoperability boundary

## Verified published standards

Verified from official publications on **2026-09-29**, before implementation:

| Standard | Version/status used | Official source |
|---|---|---|
| FHIR | R4 4.0.1, published; retained for the existing normalized model and ABDM compatibility | https://hl7.org/fhir/R4/ |
| SMART App Launch | 2.2.0, published STU | https://hl7.org/fhir/smart-app-launch/STU2.2/app-launch.html |
| CDS Hooks | 2.0.1, published STU2 | https://cds-hooks.hl7.org/STU2/ |
| patient-view | 1.0, hook catalog maturity 5 | https://cds-hooks.org/hooks/patient-view/ |
| ABDM FHIR IG | 6.5.0 published guide, FHIR 4.0.1 | https://www.nrces.in/ndhm/fhir/r4/index.html |

FHIR R4 is an intentional compatibility baseline; this does not claim it is the newest FHIR family release. No CI build is used as a released baseline. The cited ABDM published package remains 6.5.0; a 7.0.0 preview is not used, and its release status was not independently established by an accessible publication during verification. Some individual 6.5.0 resources carry `draft` metadata; that is distinct from the version of the published guide.

## SMART public client with PKCE

Register the application as a public SMART client, with an exact redirect URI. Configure:

```dotenv
SMART_CLIENT_ID=your-registered-public-client-id
SMART_REDIRECT_URI=http://localhost:18000/smart/callback
SMART_ISSUERS=["https://your-synthetic-sandbox.example/fhir"]
SMART_ENDPOINT_ORIGINS=["https://your-synthetic-sandbox.example"]
SMART_STATE_SECONDS=300
```

The HTTP loopback callback is permitted only in development. Production requires HTTPS. Issuers must be exact configured URLs; OAuth origins must be registered. Discovery uses `{iss}/.well-known/smart-configuration`; unexpected authorization/token endpoints, query-bearing endpoints, and missing S256 support are rejected. The backend uses a DNS-pinned, TLS-verified HTTPS transport with no redirects or proxy environment inheritance.

Flow: EHR opens `/smart/launch?iss=...&launch=...`; backend discovers metadata, creates cryptographically random state, PKCE verifier/challenge, and browser binding; it redirects to authorization. A HttpOnly SameSite=Lax cookie binds the callback to that browser. PostgreSQL stores hashes of state/browser binding and a short-lived verifier. `/smart/callback` locks and consumes state once before token exchange. Expired, mismatched, or replayed state fails closed. Token exchange includes the original verifier and configured redirect URI.

Scopes are `launch patient/Patient.r` plus explicit `patient/Condition.rs`, `patient/Observation.rs`, `patient/MedicationRequest.rs`, `patient/AllergyIntolerance.rs`, `patient/Encounter.rs`, and `patient/Procedure.rs`. There are no patient/user/system wildcard scopes, refresh access, or client secret. The authorization server validates PKCE; integration tests implement that validation in a controlled OAuth server.

The token response's patient is authoritative. No conflicting browser patient selection is accepted. The server fetches Patient and the six supported search resources with Bearer authorization, at most ten pages per resource and 1000 resources overall. Pagination must remain on the registered server/base path; cycles, malformed bundles, cross-patient resources and unsupported types fail closed. A 401 is an acquisition failure, not empty clinical data.

Only normalized synthetic context is retained briefly for handoff. The browser receives a non-secret handoff hash and imports through authorized `/v1/smart/{handle}/import`; the transient context is consumed. Tokens, refresh tokens and authorization codes are never stored in browser persistence or operational audit. Uvicorn access logging is disabled because callback URLs contain codes; any reverse proxy must also omit query strings. SMART establishes patient context, not application clinician identity.

### Sandbox validation

Offline tests use controlled discovery, OAuth and FHIR transport: successful launch/import, wrong/expired/reused state, wrong browser, PKCE rejection, token failure, missing patient, malformed FHIR, private destinations and pagination failures. CI needs no public sandbox. For a manual SMART Health IT test, register the exact public-client configuration, configure its published HTTPS FHIR/auth origins, choose synthetic records, and launch from the sandbox. The strict synthetic identifiers/codes supported by this repository may require loading its fixtures into the sandbox. No live sandbox certification or completed live OAuth smoke is claimed.

## CDS Hooks

`GET /cds-services` advertises only `medical-patient-review`, hook `patient-view`. Call `POST /cds-services/medical-patient-review` with a UUID `hookInstance`, `hook: patient-view`, and context containing `userId` (FHIR resource reference) and `patientId`; `encounterId` is optional. This follows the separately verified hook contract. `hookInstance` correlates and deduplicates a review; it is never an authentication credential.

Discovery supplies bounded prefetch templates for Patient and the six supported resources (`_count=100`). Complete prefetch is normalized directly. Missing/incomplete prefetch can use `fhirServer` and standard `fhirAuthorization`; only a configured server is accepted, and the same bounded FHIR client handles retrieval. Inbound service authentication uses the application's Bearer/JWT roles when `CDS_REQUIRE_AUTH=true` (default), and always in production. This is a deployment choice, not an invented universal CDS Hooks authentication requirement. Discovery is public.

The deterministic review engine writes a snapshot-bound review and findings. Potential gaps produce warning cards, missing-data/suppressed findings information cards, and satisfied/not-applicable findings no cards. Cards contain source links, recommendation/version references, and deterministic rationale. They never contain autonomous treatment suggestions or ordering actions. Missing record coverage remains insufficient data. `CDS_SYNTHETIC_FIXTURE_COVERAGE=true` permits an exact server-owned fixture-hash match only in development/test; it cannot be used to assert arbitrary longitudinal coverage or enabled in production.

```bash
python -m scripts.cds_demo > reports/cds-request.json
curl http://localhost:18000/cds-services
# Supply a valid application clinician JWT in CDS_TOKEN.
curl -H "Authorization: Bearer $CDS_TOKEN" -H "Content-Type: application/json" --data-binary @reports/cds-request.json http://localhost:18000/cds-services/medical-patient-review
```

The controlled Docker smoke signs an ephemeral development clinician token and exercises discovery and patient-view without an LLM. Its fixture is explicitly synthetic. There is no feedback endpoint, order-select hook, automatic action execution, or general EHR certification in this phase.
