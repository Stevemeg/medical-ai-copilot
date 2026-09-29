# Compliance considerations

This repository is a synthetic-data engineering and portfolio demonstration. It has no established regulatory clearance, certification, independent clinical validation, or authorization for real patient deployment. The software is not offered as a medical device or a replacement for clinical judgment.

Implemented controls include verified JWT/OIDC authentication, server-side roles, immutable patient snapshots/findings, separate clinician dispositions, lifecycle-governed evidence, independent claim verification, transactional PostgreSQL writes and HMAC audit, SSRF-resistant acquisition, bounded inputs, and ephemeral operational retention. These are engineering controls, not a legal compliance determination.

Production operational audit records identifiers, hashes, event types, and request correlation rather than raw questions or FHIR bodies. The old SQLite audit implementation is legacy code, not the production audit authority. Encryption at rest, TLS termination, backup restoration, deployment access controls, data residency, vendor agreements, and an external audit checkpoint anchor remain deployment responsibilities. Ephemeral rate/idempotency/SMART rows have cleanup; clinical and audit records are deliberately excluded.

ABDM work is selected published-profile compatibility analysis using fictional resources. There is no ABHA identity integration, consent-manager integration, NHA approval, or ABDM certification. SMART and CDS Hooks are development integration boundaries, not EHR certification.

Any future use with real people would require a separately defined intended use, clinical assessment, privacy and security review, operational governance, and qualified review of the applicable jurisdiction's current requirements. No conclusion about HIPAA, FDA, or other legal status can be inferred from this repository's test results.
