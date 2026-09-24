# Synthetic patient context and review

This phase adds a **FHIR R4-compatible supported subset** for fictional demo patients. It does not claim full FHIR validation or interoperability conformance. The product path is:

```text
Synthetic collection Bundle -> structural validation -> FHIR adapter
  -> immutable normalized PatientContext -> SHA-256 context hash
  -> SQLite patient snapshot -> deterministic timeline and data availability
  -> persisted ClinicalReview snapshot -> clinician workspace

Governed source registry -> lifecycle-aware evidence retrieval -> Ask Evidence
```

The patient path and Ask Evidence path are separate. Patient records are not automatically sent to the LLM. Current clinical retrieval excludes the superseded bundled NICE NG28 2022 and NG19 2019 versions. Phase 3 adds a narrow deterministic hypertension annual follow-up finding; the foot-assessment candidate is suppressed pending current governed evidence. See [clinical rules](CLINICAL_RULES.md).

## Supported import subset

Imports accept one `Bundle` with `type: collection`, exactly one `Patient`, and zero or more `Condition`, `Observation`, `MedicationRequest`, `AllergyIntolerance`, `Encounter`, and `Procedure` resources in any order. Every resource needs an `id`. Related resources need a relative `Patient/<id>` subject reference; `AllergyIntolerance` uses `patient`. Duplicate resource type/id pairs and unsupported resource types are rejected. The patient must have a synthetic marker, a `SYN-PAT-` identifier, and a `Synthetic Patient` display label. This is a development safeguard, not a means of proving that arbitrary input has no PHI.

Consumed fields include patient birth date and gender; condition code/status/onset/recorded date; numeric observation value or numeric components, code/status/effective date; medication code/status/intent/authored date/dosage text; allergy code/status/verification/category/criticality/recorded date/reaction manifestation; encounter status/class/type/period; and procedure code/status/performed date. Coding system, code, and display are preserved when supplied. Blood pressure components remain separate observations components.

This subset does not support transaction Bundles, contained resources, absolute or URN patient references, extensions, narrative, terminology validation, observation value types other than numeric Quantity, patient identifiers beyond the synthetic import marker, partial FHIR dates (year or year-month), or complete resource/profile constraints. Date-only values remain date-only. A dateTime with a time requires an offset or `Z` and is normalized to UTC. Malformed consumed fields are rejected with structured errors.

The scoped structure follows the [HL7 FHIR R4 Bundle definition](https://hl7.org/fhir/R4/bundle.html) and [R4 Observation component model](https://hl7.org/fhir/R4/observation.html). The fixture blood pressure panel and component codes follow the [R4 vital sign value set](https://hl7.org/fhir/R4/valueset-observation-vitalsignresult.html).

## Normalized context and reproducibility

Raw FHIR is accepted only at the import boundary. `PatientContext` is a typed internal model containing the patient, conditions, observations, medications, allergies, encounters, procedures, and source metadata. Resource arrays and coding arrays are sorted deterministically. SHA-256 is computed over canonical JSON of normalized content, so Bundle entry order and unused transport fields do not affect it. A meaningful normalized content change creates a new hash.

SQLite stores normalized context JSON, its hash, an independent UUID patient ID, and review JSON. Raw Bundles are not persisted. Reimporting unchanged content is idempotent. Reimporting changed content with the same synthetic source patient ID updates that patient row and keeps its UUID; existing reviews retain their copied context and original hash. Every explicit review creation makes a new review UUID. Phase 3 findings and clinician actions use separate SQLite tables. A review can be marked completed; completed reviews cannot be evaluated. Local `data/patient_context.db` is ignored by Git. SQLite provides local persistence between requests but is not a shared production storage or security system.

## Context queries and timeline

Helpers expose active conditions (`active`), active medication requests (`active`), known allergies (excluding inactive, refuted, and entered-in-error), code-based observation history, latest observation, and date-range observation queries. Entered-in-error and cancelled observations are excluded from active observation queries and timeline. These are intentionally narrow status rules; they do not represent complete FHIR workflow semantics.

The timeline combines conditions, observations, medication requests, allergies, encounters, and procedures. Dated entries sort newest first with deterministic type and resource ID ties. Undated entries follow dated entries and remain visible. No timestamp is invented. Data availability reports which resource categories were supplied, which were absent, and the last supplied observation date by display name. It makes no judgment about whether data is due, overdue, clinically sufficient, or appropriate.

## API and UI

The versioned API provides `POST /v1/fhir/validate`, `POST /v1/patients/import`, `GET /v1/patients`, `GET /v1/patients/{id}`, `POST /v1/patients/{id}/reviews`, `GET /v1/patients/{id}/reviews`, `GET /v1/reviews/{id}`, and `POST /v1/reviews/{id}/complete`. Demo fixture routes are `GET /v1/demo-patients` and `POST /v1/demo-patients/{key}/load`. `GET /v1/knowledge-sources` exposes registry lifecycle metadata. Existing `/api/ask` remains the separate Ask Evidence route.

The Streamlit workspace shows patient selection/import, summary, timeline, data availability, review snapshots, deterministic Phase 3 findings, Ask Evidence, and knowledge sources. The alternative HTML client has not yet been updated with Phase 3 finding controls. All bundled records in `data/synthetic_fhir/` are fictional. Potential care gaps require clinician review; the interface does not diagnose, prescribe, or issue autonomous treatment recommendations.

Normal application logs contain operation identifiers, counts, and hashes; they do not include raw Bundles or normalized patient payloads. The current local prototype has no authentication and must not be used with real patient data.

## Run and test

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest -q
ruff check .
ruff format --check app.py api_server.py backend/patient_*.py backend/fhir_adapter.py tests/test_patient_phase2.py scripts/generate_synthetic_fhir.py
python -m mypy
streamlit run app.py
```

For the alternative HTML client, run `uvicorn api_server:app --port 8000` and serve `frontend/index.html` locally. Generate the committed demo fixtures again with `python scripts/generate_synthetic_fhir.py`.
