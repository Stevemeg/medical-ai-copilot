# Deterministic clinical review rules

Phase 3 implements two active coded annual follow-up checks in **synthetic** records. The rules do not diagnose, prescribe, select treatment, interpret BP or HbA1c targets, or score a patient. A potential care gap is a request for clinician review, not a clinical conclusion.

## Architecture and lifecycle

```text
FHIR Bundle -> PatientContext -> immutable ClinicalReview.patient_snapshot
                                   + explicit as_of and RecordCoverage
                                   -> RuleRegistry -> evidence validation
                                   -> deterministic ClinicalReviewEngine
                                   -> immutable ClinicalFinding -> append-only FindingAction
```

`RuleDefinition` stores stable ID, semantic version, domain, status (`active`, `disabled`, `retired`), effective dates, required data, rule kind, and recommendation provenance. `RuleRegistry` rejects duplicate ID/version pairs and malformed identifiers. An active rule must pass one of two evidence paths: (A) a registered, ingested **current clinical guideline** version, or (B) a `verified_current` authoritative recommendation snapshot from `data/rule_evidence_registry.json`. Both paths require matching canonical document identity. When a rule links both a local current version and a recommendation snapshot, both must validate. Invalid evidence blocks activation. The engine checks evidence again before each new evaluation: lifecycle or verification drift suppresses the rule. A previously saved finding is preserved as historical output. A disabled rule exposes a safe `suppression_reason` code.

`RuleEvaluationContext` carries the immutable review snapshot, its hash, explicit `as_of`, and a separate `RecordCoverage` assertion. Runtime evaluation is offline, has no LLM calls, and uses no hidden clock. The API requires the caller to submit `as_of`; the Streamlit demo offers a date input.

The five finding statuses are `satisfied` (coded event within the interval), `potential_care_gap` (complete interval asserted and no timely qualifying event), `insufficient_data` (the question cannot safely be resolved), `not_applicable` (no supported active condition or patient is a minor), and `suppressed` (rule or evidence unavailable). Findings retain structured observations, missing-data identifiers, rule version, source reference, context hash, and evaluation date. IDs derive from review, rule/version, snapshot hash, and evaluation request. Evaluating the same review/request returns its existing findings; a different evaluation context requires a new review. Completed reviews cannot be evaluated.

`RecordCoverage` is demo metadata, **not a FHIR field**. It asserts a complete resource type over a start/end interval. Missing event plus missing/partial/wrong-type coverage yields `insufficient_data`, never a gap. Coverage is an external assertion and is not independently verified. It must not be inferred from the presence of a few resources. The demo assertions are in `data/synthetic_fhir/manifest.json`; imported Bundles have no asserted coverage by default.

## Rule catalogue

| Rule ID / version | Domain and purpose | Applicability and required data | Evidence | Outputs and limits |
| --- | --- | --- | --- | --- |
| `HTN_ANNUAL_CARE_REVIEW` / `1.0.0` | Hypertension; presence/timing of coded annual care review | Active ICD-10 `I10`, adult birth date, completed Encounter with project code `HTN-ANNUAL-REVIEW`, complete Encounter coverage for the calendar-year interval | NICE NG136, recommendation 1.4.24, local `nice-ng136-2026-02-26`, verified 2026-09-24 | All five statuses. Does not determine quality of the review, BP control, or treatment. |
| `DIABETES_ANNUAL_FOOT_ASSESSMENT` / `1.0.0` | Diabetes; presence/timing of annual foot-risk assessment | Active ICD-10 `E11`/`E11.9`, adult birth date, coded Procedure and complete Procedure coverage | NICE NG19, recommendation 1.3.3, verified authoritative recommendation snapshot `nice-ng19-rec-1.3.3` on 2026-09-24 | All five statuses. Does not cover risk stratification or diagnosis-time, new-foot-problem, or hospital-admission workflow. The bundled 2019 PDF is **not** this rule's evidence. |

The project event coding system is `urn:medical-ai-copilot:synthetic-clinical-event`. These are synthetic codes and are **not** SNOMED CT. Uncoded or ambiguous relevant events yield `insufficient_data`. The supported condition code set is intentionally narrow; other diabetes and hypertension codes need terminology verification before expansion. The annual lower bound is the same calendar date one year before `as_of`, inclusive; 29 February maps to 28 February in the prior year. Future and undated events cannot establish satisfaction. If the condition onset is missing or less than one year ago and no timely event exists, the check returns `insufficient_data` because an annual reassessment may not yet be due. Diagnosis-time checks are outside Phase 3.

## Evidence verification and governance

| Guideline | Official NICE check on 2026-09-24 | Local source and disposition |
| --- | --- | --- |
| [NG136 recommendations](https://www.nice.org.uk/guidance/ng136/chapter/recommendations) | Last updated 26 February 2026; 1.4.24 still concerns annual hypertension care review | Ingested `nice-ng136-2026-02-26`, current; active binding |
| [NG19 recommendations](https://www.nice.org.uk/guidance/ng19/chapter/recommendations) and [2023 evidence review](https://www.nice.org.uk/guidance/ng19/evidence/b-risk-assessment-models-and-tools-for-predicting-the-development-of-diabetic-foot-problems-and-foot-review-frequency-pdf-6953995119) | Page header still says last updated 11 October 2019, but current page contains 2023 recommendations and evidence review. Recommendation 1.3.3 still includes adult assessment at diagnosis and at least annually. [2025 surveillance](https://www.nice.org.uk/guidance/ng19/evidence/july-2025-exceptional-surveillance-of-diabetic-foot-problems-prevention-and-management-nice-guideline-ng19-15373951597) concerns ulcer treatment, not this recommendation. | Bundled `nice-ng19-2019-10-11` is not a current version of the **whole** guideline and remains `superseded`. Rule B binds instead to the separately verified current recommendation snapshot. No new local PDF version is fabricated. |
| [NG28 guidance](https://www.nice.org.uk/guidance/ng28/chapter/Blood-glucose-management) | Last updated 18 February 2026 | Bundled `nice-ng28-2022-06-29` remains superseded. Registry rejects active clinical-rule binding; there is no HbA1c rule. |

Retrieval knowledge and rule knowledge have different evidence contracts. `data/source_registry.json` governs locally ingested PDF versions, page chunks, and RAG retrieval. `data/rule_evidence_registry.json` governs specific authoritative recommendations checked when rules were authored. A current recommendation snapshot does **not** make a superseded local PDF eligible for current retrieval. NG19's 2019 PDF remains excluded from that retrieval; its recommendation 1.3.3 snapshot can support Rule B. The two registries share the canonical document ID. Evidence cards report the basis so the local PDF is never mislabeled as current. These links do **not** claim claim-level generation verification. No new NICE PDF or full recommendation text is redistributed.

### Recommendation verification procedure

The registry stores publisher, guideline code, recommendation ID, canonical official URL, jurisdiction, verification date and status, a short paraphrase, and the SHA-256 of the exact recommendation body. The fingerprint method locates the NICE HTML `article` whose ID combines guideline and recommendation number, extracts its `recommendation__body`, decodes HTML entities, normalizes Unicode to NFC, maps bullet markers to `•`, collapses whitespace, and hashes UTF-8 bytes. It does not remove substantive words or punctuation. Because HTML structure and publisher copy can change, a different hash requires human review even if the change looks cosmetic.

Run `python -m scripts.verify_rule_evidence` explicitly during governance review. It fetches only configured NICE recommendations pages over HTTPS, does not follow redirects, and reports `MATCH`, `DRIFT`, `NOT_FOUND`, or `FETCH_FAILED`. `DRIFT` and `NOT_FOUND` atomically mark the corresponding registry record `drift_detected`, so the rule fails closed on later startup/evaluation. `MATCH` never automatically restores a drifted record. On a difference, a reviewer checks the official recommendation and any surveillance, decides whether the rule still expresses it, revises rule behavior/version if needed, then updates the fingerprint, verification date, and `verified_current` status in a reviewed commit. A transient fetch failure leaves the last reviewed status unchanged and is reported as an unresolved governance check. Normal tests use synthetic HTML and are offline; patient evaluation never calls NICE.

## Human review, storage, and limitations

PostgreSQL stores findings separately from immutable review snapshots with foreign keys and unique `(review_id, rule_id, rule_version)`. Readback checks finding review ID and context hash. Clinician dispositions are append-only records with unique IDs and timestamps: accept, dismiss, already addressed, incorrect evidence, and not clinically relevant. Dispositions never rewrite system findings. API actions store the actor subject, and the clinical write and minimized audit event commit together. The review row lock prevents concurrent duplicate evaluation. See [production architecture](PRODUCTION_ARCHITECTURE.md).

The input must be synthetic. Coverage assertions can be wrong, and coded events only establish that an event was recorded, not that the care was adequate. The annual checks implement only a narrow subset of each recommendation. The app does not make treatment or diagnosis decisions. New data requires a new review; an older review continues to describe its original snapshot and rule version.
