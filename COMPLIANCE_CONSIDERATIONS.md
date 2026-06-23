# Compliance considerations

**This is an educational analysis for portfolio purposes, not legal
advice.** Real compliance sign-off for an actual clinical deployment
requires an actual lawyer or compliance officer reviewing the actual
deployment context -- this document exists to show awareness of what
that review would need to cover, not to substitute for it.

This project does not handle real patient data, is not deployed in a
clinical setting, and has no real users beyond demonstration/portfolio
purposes. The considerations below describe what WOULD be relevant if
that ever changed, mapped against what this codebase currently does.

---

## HIPAA (Health Insurance Portability and Accountability Act)

HIPAA's Security Rule requires specific technical safeguards for systems
that handle Protected Health Information (PHI) -- identifiable
information about a person's health condition, care, or payment for care.

| Safeguard | What HIPAA requires | Current state in this project |
|---|---|---|
| Access controls | Unique user identification, authentication, and authorization before PHI access | Not applicable -- this app has no login and accepts no patient-identifying input by design. If real PHI were ever introduced, this would need real authentication (see PROJECT_NOTES.md for the explored-and-deliberately-not-implemented Google OAuth path). |
| Audit controls | Hardware, software, or procedural mechanisms to record and examine activity in systems containing PHI | Implemented as a real, working mechanism (`backend/audit_log.py`) -- a tamper-evident hash-chained log of every question/answer/sources triple. Verified to detect both row modification and row deletion (see that file's docstring and the verification tests run during development). This satisfies the AUDIT MECHANISM requirement in isolation, but see "Gaps" below. |
| Transmission security | Encryption of PHI in transit | Not currently relevant -- no PHI is transmitted. If deployed with real patient data, this would require HTTPS enforcement (not currently configured; Streamlit Community Cloud provides this automatically, a custom deployment would need to add it explicitly) and encrypted connections to any external API (Groq's API is already HTTPS-only). |
| Encryption at rest | PHI stored at rest should be encrypted | The audit log (`data/audit_log.db`) is currently an unencrypted local SQLite file. If it ever logged real PHI, this would need disk-level or application-level encryption. |
| Minimum necessary | Only the minimum PHI necessary for the task should be used/disclosed | The audit log intentionally captures full question/answer text, which would be MORE than "minimum necessary" if real PHI were ever in a question. This is a real gap if scope ever expands -- see below. |

### Gaps if this were ever used with real patient data

1. **No de-identification step.** The audit log stores full question
   text verbatim. If a user ever typed real patient details into the
   question box, that would be logged in full -- there's no scrubbing or
   redaction layer. A real clinical deployment would need this before
   audit logging could be considered compliant, not after.
2. **No Business Associate Agreement (BAA) with Groq.** HIPAA requires a
   signed BAA with any third-party vendor that processes PHI on your
   behalf. This project calls Groq's API directly with no such agreement
   in place, because no PHI is sent to it under the current design --
   but this would be a hard blocker for real clinical use without one.
3. **No encryption at rest** for the audit log, as noted above.

---

## FDA Clinical Decision Support (CDS) guidance

The FDA's guidance on Clinical Decision Support software (under the 21st
Century Cures Act's Section 520(o)(1)(E) software function exclusion)
distinguishes between CDS tools that are explicitly excluded from FDA
device regulation and those that aren't, based on four statutory
criteria. **This guidance was updated twice in early 2026** (a January 6,
2026 update, itself superseded by a further revision on January 29,
2026), replacing the 2022 guidance that many existing write-ups online
still describe -- a concrete example of why this section needs to be
re-checked against whatever the current guidance says at the time of any
real decision, not treated as a fixed reference.

**The four non-device CDS criteria** (unchanged by the 2026 updates,
per FDA's own town hall materials on the revision) require that the
software:
1. Does not acquire, process, or analyze a medical image, an in vitro
   diagnostic device signal, or a signal-acquisition-system pattern
2. Displays, analyzes, or prints medical information about a patient
3. Supports or provides recommendations to a healthcare provider about
   prevention, diagnosis, or treatment
4. Enables the provider to independently review the basis for the
   recommendations, rather than primarily relying on the software's
   judgment

**Where this project currently sits**: it satisfies (1) trivially (no
image/signal processing at all). It's designed around (4) directly --
every answer is grounded in cited, page-referenced source guidelines a
physician could independently check, which is the core design intent
behind the citation system built earlier in this project. It's
informational/educational by explicit framing (the in-app disclaimer
states this directly), not a tool that makes patient-specific diagnostic
or treatment decisions about an actual identified patient.

**A specific nuance worth flagging**: earlier (2022-era) FDA practice
treated software providing only a SINGLE recommendation more strictly
than software presenting multiple options for a clinician to weigh --
some legal commentary describes the 2026 revision as softening that
distinction. Since this tool sometimes returns a single direct answer
rather than multiple options, this is exactly the kind of detail that
would need checking against the actual current guidance text by someone
qualified to interpret it, not inferred from a summary like this one.

**What would need real legal review before any clinical claim**: whether
this tool's positioning crosses from "informational reference" into
regulated "clinical decision support" depends on exact phrasing, intended
use claims, and how it's marketed -- this is exactly the kind of judgment
call that needs an actual regulatory attorney working from the current
guidance text, not a self-assessment. This document is not that
assessment.

---

## Data residency and retention

Not currently configured at all -- the audit log has no retention policy
(it grows indefinitely) and no data residency controls (it's a local
file with no geographic guarantees). A real deployment subject to
data-residency requirements (e.g. GDPR's requirements for EU resident
data) would need explicit retention limits and documented data location
guarantees, neither of which exist here.

---

## Summary

This project demonstrates the *pattern* of an audit trail (real,
verified, tamper-evident) and an awareness of what HIPAA/FDA
considerations would apply -- but it is not, and does not claim to be,
compliant with either framework for actual clinical use. The honest gaps
(no de-identification, no BAA, no encryption at rest, no real auth, no
retention policy) are listed above specifically so they aren't quietly
glossed over.