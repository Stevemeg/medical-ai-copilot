# ABDM profile compatibility analysis

The published official baseline independently checked on **2026-09-29** is **ABDM FHIR IG 6.5.0**, FHIR **4.0.1**, package **ndhm.in#6.5.0**. This is ABDM-aligned mapping and selected-profile analysis, **not certification, ABHA integration, NHA approval, or a production ABDM exchange**.

Official guide: https://www.nrces.in/ndhm/fhir/r4/index.html

Official package: https://www.nrces.in/ndhm/fhir/r4/package.tgz

Downloaded package SHA-256: `41fa99955adbd1983a235d7c7136b30f423ce34ff541d86b354d95b7704527d4`. Eight exact official StructureDefinitions are vendored under `data/abdm/`, with per-file checksums, package ID/version and CC0 license recorded in `manifest.json`. Tests reject drift. `scripts.vendor_abdm_profiles` verifies the pinned package checksum and reads only named, bounded archive members; it does not extract arbitrary paths. No mutable `latest` dependency is used.

The 7.0.0 preview mentioned in earlier planning is not the baseline. No 7.0.0 constraints are mixed into this analysis; an accessible official published 7.0.0 release was not established at verification. See INTEROPERABILITY.md for the published/draft distinction.

## Mapping

Profile URLs below are the official canonical URLs in the package, distinct from the guide's `www` download host.

| Internal collection | R4 / official ABDM profile | Supported fields | Required-field and terminology gaps | Status |
|---|---|---|---|---|
| patient | Patient / https://nrces.in/ndhm/fhir/r4/StructureDefinition/Patient | Synthetic ID, label, birth date, gender | Official identifier cardinality; normalized model does not preserve full identifier types/systems, telecom/address; no ABHA | Partial |
| conditions | Condition / https://nrces.in/ndhm/fhir/r4/StructureDefinition/Condition | Code, subject, clinical/verification status, onset | Code and subject required; full SNOMED membership, categories and nested constraints not validated | Partial |
| observations | Observation / https://nrces.in/ndhm/fhir/r4/StructureDefinition/Observation | Status, code, subject, effective date, Quantity/components | Status/code required; limited LOINC/UCUM normalization, no full value-set membership or other value types | Partial |
| blood pressure observations | Observation / https://nrces.in/ndhm/fhir/r4/StructureDefinition/ObservationVitalSigns | Numeric BP components and units | Vital-sign category, specific component slicing and terminology not fully exported/validated | Partial, analysis only |
| medications | MedicationRequest / https://nrces.in/ndhm/fhir/r4/StructureDefinition/MedicationRequest | Medication code, subject, status, authored date | Official intent, medication choice, requester and dosageInstruction requirements; internal model lacks full requester/dosage representation | Incomplete |
| allergies | AllergyIntolerance / https://nrces.in/ndhm/fhir/r4/StructureDefinition/AllergyIntolerance | Code, patient, clinical and verification status | Code/patient required; terminology, reaction detail and target-profile validation incomplete | Partial |
| encounters | Encounter / https://nrces.in/ndhm/fhir/r4/StructureDefinition/Encounter | Status, subject, type, period | Status/class required; internal encounter class not retained; full context and terminology incomplete | Incomplete |
| procedures | Procedure / https://nrces.in/ndhm/fhir/r4/StructureDefinition/Procedure | Status, subject, code, performed date | Status/code/subject required; full SNOMED and performer/reference-profile constraints unvalidated | Partial |

The importer remains a scoped synthetic R4 adapter, not a generic ABDM converter. It does not add `meta.profile` and claim conformance. Normalized fields can be fewer than the raw resource's fields. Unsupported resource types, terminology and document composition are reported rather than fabricated. No Composition, document Bundle workflow, consent artifact, HIP/HIU exchange or longitudinal ABHA linkage is implemented.

## Reproducible validation

`python -m scripts.validate_abdm` checks the package/profile hashes and analyzes all 33 synthetic fixture resources against direct required fields from the official profile snapshots. `eval/abdm_report.json` records actual missing required direct fields, terminology bindings and explicitly unvalidated nested cardinality, slicing, FHIRPath invariants, terminology membership and target profiles. The runner does not describe a lack of direct-field errors as full profile conformance.

The fixtures use synthetic labels and identifiers only; no real ABHA identifiers are present. Negative tests intentionally omit required identifiers to prove that merely naming a profile cannot make a resource valid. This limited, transparent compatibility report is the Phase 6 ABDM mapping artifact. Full deployment validation requires the HL7 validator plus the exact guide dependencies and a configured terminology service, followed by the broader ABDM integration requirements; no external certification is inferred.

## Official HL7 validator run

The synthetic Patient example in `data/abdm/examples/patient.json` was additionally run through **HL7 validator 6.10.4**, Java24, against the pinned official 6.5.0 package. The first run found a missing nested identifier coding display (and rejected the synthetic example URL under its default policy). The corrected example includes the required display and narrative; synthetic example URLs were explicitly permitted for that run.

`eval/abdm_patient_validation.json` is the unedited OperationOutcome: **0 errors, 0 warnings, 1 informational notice** for the private synthetic tag code system. `eval/abdm_patient_negative_validation.json` preserves the original failed outcome. `eval/abdm_validation_manifest.json` records the validator JAR checksum, exact guide checksum, loaded dependency versions, archived input/output checksums and issue counts. The earlier input is archived as `patient_incomplete.json`; the original execution used the same working Patient filename before correction.

The command was `java -Xmx2g -jar reports/tools/validator_cli.jar data/abdm/examples/patient.json -version 4.0.1 -ig reports/abdm-package.tgz -profile https://nrces.in/ndhm/fhir/r4/StructureDefinition/Patient -tx n/a -allow-example-urls true -output eval/abdm_patient_validation.json`. This validates the selected profile's structure and invariants, with external terminology disabled. It does not establish terminology conformance, other resources' full conformance, or ABDM certification. The full validator is an opt-in development check; offline CI verifies the pinned profiles, direct-cardinality regression and recorded artifact checksums. The validator's ancillary terminology/extension packages were resolved at installation and are recorded in the manifest; only the ABDM release itself is the mapping baseline.
