# Primary web workspace

Open http://localhost:18000/ after `docker compose up -d --build --wait`. FastAPI serves `frontend/index.html` and same-origin assets; no separate static server or hard-coded API origin is needed.

Patient Review is primary, with saved Review Findings, Ask Evidence, Guideline Intelligence and System Status navigation. The UI shows snapshot identity, observations/timeline, missing data, deterministic rationale, rule evidence and append-only dispositions. An explicit date and completeness assertion is required to establish a care gap; data fetch success does not establish coverage.

The optional session field accepts an existing authorized JWT and holds it in memory only. Development anonymous clinician mode cannot activate guidelines. Authenticated guideline_editor/clinical_admin roles reveal governance controls; server-side authorization applies independently. Production identity-provider login UI is outside this demo; supply a valid deployment token through the session boundary.

SMART completion imports the browser-bound handoff and labels the context `SMART launch / FHIR server`; the access token is never exposed to the UI. Keyboard-operable controls, labels, headings, text statuses, accessible colors and explicit historical warnings support basic accessibility. The DOM smoke is `node eval/ui_smoke.js`; it is not a full accessibility certification.
