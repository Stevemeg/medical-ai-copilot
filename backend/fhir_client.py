"""Bounded patient-scoped FHIR reads shared by SMART and CDS Hooks."""

from urllib.parse import urlencode, urljoin

from backend.fhir_adapter import FHIR_ID, SUPPORTED_TYPES, parse_bundle
from backend.safe_http import AcquisitionError, SafeHTTP, origin

RESOURCE_TYPES = tuple(sorted(SUPPORTED_TYPES - {"Patient"}))


class FHIRClient:
    def __init__(self, base: str, token: str, http: SafeHTTP):
        origin(base)
        self.base = base.rstrip("/")
        self.token = token
        self.http = http

    def get(self, url: str) -> dict:
        if not url.startswith(self.base + "/") or origin(url) != origin(self.base):
            raise AcquisitionError("FHIR pagination escaped registered base")
        return self.http.json(url, headers={"Authorization": f"Bearer {self.token}"})

    def bundle(self, patient_id: str) -> dict:
        if not FHIR_ID.fullmatch(patient_id):
            raise AcquisitionError("Invalid FHIR patient context")
        patient = self.get(f"{self.base}/Patient/{patient_id}")
        if patient.get("resourceType") != "Patient" or patient.get("id") != patient_id:
            raise AcquisitionError("FHIR patient context mismatch")
        resources = [patient]
        for resource_type in RESOURCE_TYPES:
            url = f"{self.base}/{resource_type}?" + urlencode({"patient": patient_id, "_count": 100})
            visited = set()
            for _ in range(10):
                if url in visited:
                    raise AcquisitionError("FHIR pagination cycle")
                visited.add(url)
                page = self.get(url)
                if page.get("resourceType") != "Bundle" or page.get("type") != "searchset":
                    raise AcquisitionError("Expected FHIR searchset")
                entries = page.get("entry", [])
                if not isinstance(entries, list) or len(entries) > 100:
                    raise AcquisitionError("FHIR page exceeds supported size")
                for entry in entries:
                    resource = entry.get("resource", {})
                    if resource.get("resourceType") != resource_type:
                        raise AcquisitionError("Unexpected FHIR resource")
                    resources.append(resource)
                if len(resources) > 1000:
                    raise AcquisitionError("FHIR record exceeds supported size")
                next_links = [link["url"] for link in page.get("link", []) if link.get("relation") == "next"]
                if not next_links:
                    break
                if len(next_links) != 1:
                    raise AcquisitionError("Ambiguous FHIR pagination")
                url = urljoin(self.base + "/", next_links[0])
            else:
                raise AcquisitionError("FHIR pagination limit exceeded")
        bundle = {"resourceType": "Bundle", "type": "collection", "entry": [{"resource": r} for r in resources]}
        parse_bundle(bundle)  # Enforces synthetic-only and same-patient relationships.
        return bundle
