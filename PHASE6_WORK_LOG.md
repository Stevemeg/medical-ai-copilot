# Phase 6 engineering record

Starting branch: `main`. Starting local and remote commit:
`229aefdfc0f523870f8a41f0a840de66952cc8d4`.
Fetched and fast-forward synchronized before changes; working tree was clean.
Verification date: 2026-09-29.

Inspected the application, HTML/Streamlit clients, patient and review repositories,
rules, evidence pipeline, governance registries and artifacts, migrations,
deployment configuration, evaluation scripts, existing tests and architecture docs.
The two active rules require explicit record coverage; successful FHIR searches
must never be interpreted as an assertion of longitudinal completeness.

Official publication checks before implementation:

| Standard | Version | Publication status | Official source |
|---|---|---|---|
| FHIR | 4.0.1 | R4 published; retained for repository compatibility | https://hl7.org/fhir/R4/ |
| SMART App Launch | 2.2.0 | Current published STU 2.2 | https://hl7.org/fhir/smart-app-launch/STU2.2/app-launch.html |
| CDS Hooks | 2.0.1 | Current published STU2 | https://cds-hooks.hl7.org/STU2/ |
| patient-view | 1.0 | Hook maturity 5 | https://cds-hooks.org/hooks/patient-view/ |
| ABDM IG | 6.5.0 | Current published release (individual resources may have draft status) | https://www.nrces.in/ndhm/fhir/r4/index.html |

Continuous builds and preview versions are not the conformance baseline.
Completion is conditional on the final pushed commit passing actual CI.

Final local regression: 219 tests passed with cached models and real PostgreSQL; no skips. Critical PostgreSQL subset: 22 passed. Official HL7 Patient validation against ABDM6.5.0: no errors/warnings, one synthetic-code informational notice, external terminology disabled. Rebuilt Compose patient/evidence/audit/CDS smoke passed. Local Gitleaks history and source scans found no secrets.

Clean-checkout verification identified automatic Git line-ending conversion of checksummed generated JSON. `.gitattributes` now preserves exact validated bytes for governed data and ABDM validator artifacts. This changes stored line endings where necessary, not medical text, lifecycle labels, source PDFs, vectors or threshold behavior. Fresh checkout provenance is a release gate.
