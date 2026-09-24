"""Explicit online governance check; never called during patient evaluation.

Run: python -m scripts.verify_rule_evidence
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import requests

from backend.rule_evidence import (
    REGISTRY_PATH,
    RecommendationEvidenceRegistry,
    VerificationStatus,
    extract_recommendation,
    recommendation_fingerprint,
    validate_nice_url,
)


def fetch_official_nice(url: str) -> str:
    response = requests.get(
        url, timeout=25, allow_redirects=False, headers={"User-Agent": "medical-ai-copilot-evidence-verifier/1.0"}
    )
    response.raise_for_status()
    if response.is_redirect:
        raise ValueError("redirects are not followed by the evidence verifier")
    return response.text


def verify_registry(path: Path = REGISTRY_PATH, fetcher: Callable[[str], str] = fetch_official_nice) -> dict[str, str]:
    """Compare official content, persisting a fail-closed drift state on mismatch or removal."""
    registry = RecommendationEvidenceRegistry(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = {row["evidence_id"]: row for row in payload["recommendations"]}
    outcomes: dict[str, str] = {}
    changed = False
    for entry in registry.evidence.values():
        try:
            validate_nice_url(entry.canonical_source_url, entry.guideline_code)
            html = fetcher(entry.canonical_source_url)
        except (requests.RequestException, OSError, ValueError) as exc:
            outcomes[entry.evidence_id] = f"FETCH_FAILED: {type(exc).__name__}"
            continue
        content = extract_recommendation(html, entry.guideline_code, entry.recommendation_id)
        if content is None:
            outcomes[entry.evidence_id] = "NOT_FOUND"
        elif recommendation_fingerprint(content) != entry.recommendation_sha256:
            outcomes[entry.evidence_id] = "DRIFT"
        else:
            outcomes[entry.evidence_id] = "MATCH"
        if (
            outcomes[entry.evidence_id] in {"DRIFT", "NOT_FOUND"}
            and entry.verification_status is not VerificationStatus.DRIFT_DETECTED
        ):
            rows[entry.evidence_id]["verification_status"] = VerificationStatus.DRIFT_DETECTED.value
            changed = True
    if changed:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        temporary.replace(path)
    return outcomes


def main() -> int:
    outcomes = verify_registry()
    for evidence_id, status in outcomes.items():
        print(f"{evidence_id}: {status}")
    return 0 if outcomes and all(status == "MATCH" for status in outcomes.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
