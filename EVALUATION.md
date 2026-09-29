# Evaluation and quality gates

## Datasets and provenance

`eval/cases/v1` contains versioned JSONL. Each case records case_id, suite, description, input, expected behavior, tags, provenance, version and split. Clinical query labels reference the governed repository's document/recommendation evidence. Synthetic NLI claims are explicitly synthetic, not invented publisher recommendations.

- Retrieval: 42 cases, 17 development and 25 held out. Covers hypertension, diabetes/foot historical traps, lipid/CVD, source/recommendation IDs, jurisdiction, anatomy/reference, ambiguous and out-of-domain input. Historical NG19/NG28 labels never mean current eligibility.
- Grounding: 34 cases, 11 development and 23 held out. Support/paraphrase, contradiction, near misses, numeric/interval/population changes, uncertain passages, invalid ID, stale citation, jurisdiction tampering and verifier failure.
- Clinical rules: 30 deterministic fixtures across two active rules, including exact interval/date/coverage boundaries, missing observations/age, wrong condition, stale/drifted evidence, satisfied/gap/insufficient/not-applicable/suppressed outcomes.
- Prompt injection: 12 cases, 6 development and 6 held out. A deliberately hostile fake generator repeats the retrieved attack; the evaluator inspects the final public answer for abstention, no surviving hostile claim, and no fabricated citation. This tests the final boundary under a controlled attack, not all possible live model behavior.
- `contracts.jsonl` maps additional versioned cases to executable, parameterized tests for citation, abstention, conflict, stale/historical policy, FHIR normalization, SMART, CDS Hooks, diff/activation, RBAC, SSRF, retention, concurrency/audit, ABDM and performance caching. These fixed deterministic cases use `regression` split; there is no threshold fitting or assertion that they are unseen clinical data. Reports distinguish catalog cases from expanded parameterized scenarios.

There are no duplicated retrieval queries across development/held-out. The existing pool size 30, acceptance rule and NLI thresholds were retained; Phase 6 held-out labels were not used to tune them. Development results are separate from held-out measurements. Do not call a post-tuning held-out run unbiased validation; introduce a new unseen version if tuning uses those examples.

## Commands

```bash
python -m eval.run_all --output reports/offline.json
python -m scripts.cache_models
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m eval.run_all --models --output reports/models.json
# Use a migrated, dedicated PostgreSQL test database and consistent test audit key.
PHASE5_TEST_DATABASE_URL=postgresql+psycopg://... python -m eval.run_all --models --postgres
```

The default runs deterministic rules, final-output injection, failure safety and offline test contracts. `--models` uses local pinned embedding/reranker/NLI models and never a paid generator. Model caching is an explicit installation step, separate from offline evaluation. `--postgres` includes real PostgreSQL contract tests; it refuses missing DB configuration. Skipped selected contract cases fail the evaluation. CI's PostgreSQL job also runs the full database suite.

JSON records dataset/baseline versions, counts, timestamp, source SHA, dirty-tree flag, Python/OS/CPU, per-case outcomes, thresholds, gate failures and metrics. `--release` refuses a dirty tree. The process exits nonzero when any required gate fails. Reports are written by executed code, not edited to change outcomes.

## Metrics

Document Recall@5/10, MRR and nDCG@10 use answerable queries and collapse duplicate documents. Evidence-unit Recall@5 and MRR use the explicitly labeled subset, not every query. Answerable/unanswerable counts, historical count and stale traps are separate. Abstention is evaluated on unanswerable queries; false-answer rate must remain zero on this curated safety set. Source-specific queries are intentionally part of coverage, so perfect document recall here should not be generalized.

Grounding reports supported precision/recall, nonsupport detection, false-supported and false-unsupported rates, uncertainty, and claim/evidence citation precision/recall. Wrong/stale IDs and altered jurisdiction are checked before semantics. Citation metrics assess correctness among surviving supported claims, not whether every possible source was cited. Rule findings use deterministic expected labels; precision/recall and missing-data accuracy are synthetic engineering measures.

Stage timings measure warm local retrieval (development warms models before held-out), with p50/p95 for dense, BM25, reranking and full retrieval, plus verification. They include the stated environment and corpus. SMART/CDS contract durations include local test setup and mock I/O, and are not production EHR latency estimates.

## Baseline governance

`eval/baselines/v1.json` is the initial approved engineering baseline: deterministic safety/contracts/rules/injection must all pass; held-out document Recall@5 cannot fall below the established 1.0 internal baseline; false answers/false-supported claims remain zero on these labeled safety cases. Evidence-unit ranking metrics are reported honestly, with no claim of perfect retrieval. Do not trade false-supported safety for answer rate.

For a baseline update: document the reason, prior/new per-case errors, dataset version and development-only tuning record in a pull request; run both old and proposed baselines; request independent owner review; retain the previous baseline version. PR CI reads the baseline from the base commit rather than silently trusting a changed threshold file in the PR. The initial Phase 6 bootstrap has no prior baseline and is explicitly handled. A repository owner can still change workflow code or push directly: branch protection and review are necessary organizational controls, not guarantees supplied by a JSON file.

## Release provenance

After implementation is committed and the tree is clean, execute `python -m eval.run_all --models --postgres --release`, then `python -m scripts.release_check --write-manifest --release`. Commit only the generated report/manifest as an attestation commit. The report/manifest record the exact evaluated **source commit**, and a normalized tracked-source digest excludes only these self-referential generated artifacts and the final written report. This avoids claiming a file can contain its own final commit hash. The release checker verifies the source digest, ancestor SHA, registry/index/package checksums and successful clean model report. GitHub Actions separately evaluates the final pushed SHA and uploads that run's report.

## CI and branch protection

`.github/workflows/quality.yml` runs on PR and main: `tests`, `postgres`, `evaluation`, `security`, `docker`. Python 3.11, cached dependencies, real PostgreSQL16, migration up/down/up/check, offline models, Bandit, pip-audit with reviewed exceptions, Gitleaks and Compose smoke. No application production secrets are required; test signing keys are ephemeral. Artifacts contain synthetic results and coverage only.

In GitHub Settings -> Branches -> rule for `main`, enable Require a pull request, Require approvals, Require status checks, Require branch up to date, and select the five named jobs above. Disallow force pushes/deletions and enable administrator enforcement where available. Branch protection was observed disabled during Phase 6 inspection; it is not claimed enabled. Workflow checks can be green without merge protection being enforced.
