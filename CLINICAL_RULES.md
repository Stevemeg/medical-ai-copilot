# Deterministic clinical review rules

Phase 3 implements one active coded annual follow-up check and one suppressed candidate in **synthetic** records. The rules do not diagnose, prescribe, select treatment, interpret BP or HbA1c targets, or score a patient. A potential care gap is a request for clinician review, not a clinical conclusion.

## Architecture and lifecycle

```text
FHIR Bundle -> PatientContext -> immutable ClinicalReview.patient_snapshot
                                   + explicit as_of and RecordCoverage
                                   -> RuleRegistry -> evidence validation
                                   -> deterministic ClinicalReviewEngine
                                   -> immutable ClinicalFinding -> append-only FindingAction
```

`RuleDefinition` stores stable ID, semantic version, domain, status (`active`, `disabled`, `retired`), effective dates, required data, rule kind, and recommendation provenance. `RuleRegistry` rejects duplicate ID/version pairs and malformed identifiers. An active rule needs a registered, ingested **current clinical guideline** with a registered version and canonical URL. Invalid evidence blocks activation. The engine checks evidence again before each new evaluation: lifecycle drift suppresses the rule. A previously saved finding is preserved as historical output.

`RuleEvaluationContext` carries the immutable review snapshot, its hash, explicit `as_of`, and a separate `RecordCoverage` assertion. Runtime evaluation is offline, has no LLM calls, and uses no hidden clock. The API requires the caller to submit `as_of`; the Streamlit demo offers a date input.

The five finding statuses are `satisfied` (coded event within the interval), `potential_care_gap` (complete interval asserted and no timely qualifying event), `insufficient_data` (the question cannot safely be resolved), `not_applicable` (no supported active condition or patient is a minor), and `suppressed` (rule or evidence unavailable). Findings retain structured observations, missing-data identifiers, rule version, source reference, context hash, and evaluation date. IDs derive from review, rule/version, snapshot hash, and evaluation request. Evaluating the same review/request returns its existing findings; a different evaluation context requires a new review. Completed reviews cannot be evaluated.

`RecordCoverage` is demo metadata, **not a FHIR field**. It asserts a complete resource type over a start/end interval. Missing event plus missing/partial/wrong-type coverage yields `insufficient_data`, never a gap. Coverage is an external assertion and is not independently verified. It must not be inferred from the presence of a few resources. The demo assertions are in `data/synthetic_fhir/manifest.json`; imported Bundles have no asserted coverage by default.

## Rule catalogue

| Rule ID / version | Domain and purpose | Applicability and required data | Evidence | Outputs and limits |
| --- | --- | --- | --- | --- |
| `HTN_ANNUAL_CARE_REVIEW` / `1.0.0` | Hypertension; presence/timing of coded annual care review | Active ICD-10 `I10`, adult birth date, completed Encounter with project code `HTN-ANNUAL-REVIEW`, complete Encounter coverage for the calendar-year interval | NICE NG136, recommendation 1.4.24, local `nice-ng136-2026-02-26`, verified 2026-09-24 | All five statuses. Does not determine quality of the review, BP control, or treatment. |
| `DIABETES_ANNUAL_FOOT_ASSESSMENT` / `1.0.0` | **Disabled candidate**; presence/timing of annual foot-risk assessment | Designed for active ICD-10 `E11`/`E11.9`, adult birth date, coded Procedure and complete Procedure coverage | NICE NG19, recommendation 1.3.3; bundled `nice-ng19-2019-10-11` is **superseded** | Runtime output is `suppressed`. Unit tests exercise the inactive evaluator's possible statuses, but no patient review runs it while evidence is stale. Does not cover risk stratification or diagnosis-time, new-foot-problem, or hospital-admission workflow. |

The project event coding system is `urn:medical-ai-copilot:synthetic-clinical-event`. These are synthetic codes and are **not** SNOMED CT. Uncoded or ambiguous relevant events yield `insufficient_data`. The supported condition code set is intentionally narrow; other diabetes and hypertension codes need terminology verification before expansion. The annual lower bound is the same calendar date one year before `as_of`, inclusive; 29 February maps to 28 February in the prior year. Future and undated events cannot establish satisfaction. If the condition onset is missing or less than one year ago and no timely event exists, the check returns `insufficient_data` because an annual reassessment may not yet be due. Diagnosis-time checks are outside Phase 3.

## Evidence verification and governance

| Guideline | Official NICE check on 2026-09-24 | Local source and disposition |
| --- | --- | --- |
| [NG136 recommendations](https://www.nice.org.uk/guidance/ng136/chapter/recommendations) | Last updated 26 February 2026; 1.4.24 still concerns annual hypertension care review | Ingested `nice-ng136-2026-02-26`, current; active binding |
| [NG19 recommendations](https://www.nice.org.uk/guidance/ng19/chapter/recommendations) and [2023 evidence review](https://www.nice.org.uk/guidance/ng19/evidence/b-risk-assessment-models-and-tools-for-predicting-the-development-of-diabetic-foot-problems-and-foot-review-frequency-pdf-6953995119) | Page header still says last updated 11 October 2019, but current page contains 2023 recommendations and evidence review. Recommendation 1.3.3 still has the annual adult assessment meaning. [2025 surveillance](https://www.nice.org.uk/guidance/ng19/evidence/july-2025-exceptional-surveillance-of-diabetic-foot-problems-prevention-and-management-nice-guideline-ng19-15373951597) concerns ulcer treatment. | Bundled `nice-ng19-2019-10-11` extracted 1.3.3 matches the relevant wording, but it is not a current version of the **whole** guideline. Its registry lifecycle is corrected to `superseded`; Rule B fails closed. No new local version is fabricated. |
| [NG28 guidance](https://www.nice.org.uk/guidance/ng28/chapter/Blood-glucose-management) | Last updated 18 February 2026 | Bundled `nice-ng28-2022-06-29` remains superseded. Registry rejects active clinical-rule binding; there is no HbA1c rule. |

Rule evidence references identify the official recommendation and local source version. They do **not** claim claim-level verification. The local PDF bytes are checked against the registry SHA-256 when loaded. No new NICE PDF is redistributed.

## Human review, storage, and limitations

SQLite stores findings separately from the review snapshot with foreign keys and unique `(review_id, rule_id, rule_version)`. Readback checks finding review ID and context hash. Clinician dispositions are append-only records with unique IDs and timestamps: accept, dismiss, already addressed, incorrect evidence, and not clinically relevant. Dispositions never rewrite system findings. Actions are unauthenticated demo actions; no verified actor identity is stored. Only non-sensitive IDs/statuses are logged. SQLite remains a local prototype store with limited concurrency/audit guarantees.

The input must be synthetic. Coverage assertions can be wrong, and coded events only establish that an event was recorded, not that the care was adequate. The annual checks implement only a narrow subset of each recommendation. The app does not make treatment or diagnosis decisions. New data requires a new review; an older review continues to describe its original snapshot and rule version.
